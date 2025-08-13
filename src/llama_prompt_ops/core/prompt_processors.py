# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Prompt processing chain of responsibility implementation.

This module contains the base PromptProcessor class and specific processors
that can be chained together to process prompts in a flexible, composable way.
"""

from typing import Any, Dict, List, Optional

from .utils.llama_utils import (
    format_prompt_for_llama,
    get_task_type_from_prompt,
    select_instruction_preference,
)
from .utils.logging import get_logger
import os


class PromptProcessor:
    """
    Base class for prompt processors in a chain of responsibility pattern.

    Each processor can modify the prompt data and then pass it to the next
    processor in the chain.
    """

    def __init__(self, next_processor: Optional["PromptProcessor"] = None):
        """
        Initialize a prompt processor.

        Args:
            next_processor: The next processor in the chain
        """
        self.next = next_processor

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the prompt data and pass it to the next processor.

        Args:
            data: The prompt data to process

        Returns:
            The processed prompt data
        """
        # Base implementation just passes to the next processor
        if self.next:
            return self.next.process(data)
        return data

    def set_next(self, processor: "PromptProcessor") -> "PromptProcessor":
        """
        Set the next processor in the chain.

        Args:
            processor: The next processor

        Returns:
            The next processor for method chaining
        """
        self.next = processor
        return processor


class LlamaFormatting(PromptProcessor):
    """
    Processor that applies Llama-specific formatting to prompts.
    """

    def __init__(
        self,
        next_processor: Optional[PromptProcessor] = None,
        apply_templates: bool = True,
    ):
        """
        Initialize the Llama formatting processor.

        Args:
            next_processor: The next processor in the chain
            apply_templates: Whether to apply Llama-specific templates
        """
        super().__init__(next_processor)
        self.apply_templates = apply_templates

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Llama-specific formatting to the prompt.

        Args:
            data: The prompt data to process

        Returns:
            The processed prompt data
        """
        # Skip processing if formatting is disabled
        if not data.get("apply_formatting", True):
            return super().process(data)

        # Extract examples from prompt_data if available
        examples = data.get("examples", [])
        context = data.get("context", "")
        instruction = data.get("text", "")

        # Apply Llama-specific template formatting if enabled
        if self.apply_templates:
            formatted_prompt = format_prompt_for_llama(
                instruction=instruction, context=context, examples=examples
            )
            data["text"] = formatted_prompt

        # Pass to the next processor
        return super().process(data)


class InstructionPreference(PromptProcessor):
    """
    Processor that adds task-specific instruction preferences to prompts.
    """

    def __init__(
        self, next_processor: Optional[PromptProcessor] = None, verbose: bool = False
    ):
        """
        Initialize the instruction preference processor.

        Args:
            next_processor: The next processor in the chain
            verbose: Whether to print verbose output
        """
        super().__init__(next_processor)
        self.verbose = verbose
        self.logger = get_logger()

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add task-specific instruction preferences to the prompt.

        Args:
            data: The prompt data to process

        Returns:
            The processed prompt data
        """
        # Extract input and output fields from the prompt data
        input_fields = data.get("input_fields", [])
        output_fields = data.get("output_fields", [])
        prompt_text = data.get("text", "")

        task_type = get_task_type_from_prompt(prompt_text, input_fields, output_fields)

        # Select appropriate instruction preferences based on the task type
        selected_preferences = select_instruction_preference(task_type, data)

        data["_selected_preferences"] = selected_preferences

        if selected_preferences:
            # Add the preferences as meta-instructions for MIPROv2's proposer
            proposer_kwargs = data.get("proposer_kwargs", {}) or {}

            # Combine all preferences into a single tip string
            combined_tip = "\n".join(
                [f"{i+1}. {pref}" for i, pref in enumerate(selected_preferences)]
            )
            instruction_tip = f"Apply the following instruction formats to optimize the prompt:\n{combined_tip}"

            # Store in proposer_kwargs for the MIPROv2 proposer
            proposer_kwargs["tip"] = instruction_tip
            data["proposer_kwargs"] = proposer_kwargs

            # Also store the tip directly in the strategy for YAML export
            data["instruction_tips"] = instruction_tip

            # Log the task type and selected preferences if verbose
            if self.verbose:
                self.logger.progress(f"Task type detected: {task_type}")
                for i, pref in enumerate(selected_preferences):
                    self.logger.progress(
                        f"Selected instruction preference {i+1}: {pref[:50]}..."
                    )

        # Pass to the next processor
        return super().process(data)


class AntiCompressionYellies(PromptProcessor):
    """
    Processor that applies anti-compression techniques to fight LLM compression instinct.
    Based on Field Constrictor research showing LLMs compress 88-97% when instructed to use 50%.
    """
    
    def __init__(
        self, 
        next_processor: Optional[PromptProcessor] = None,
        enabled: bool = None,
        min_length: int = 500,
        target_length: int = 800
    ):
        """
        Initialize the anti-compression processor.
        
        Args:
            next_processor: The next processor in the chain
            enabled: Whether to apply anti-compression (None = check env var)
            min_length: Minimum required length for instructions
            target_length: Target length for instructions
        """
        super().__init__(next_processor)
        
        # Check environment variable if not explicitly set
        if enabled is None:
            self.enabled = os.getenv("ANTI_COMPRESSION_YELLIES", "false").lower() == "true"
        else:
            self.enabled = enabled
            
        self.min_length = min_length
        self.target_length = target_length
        self.logger = get_logger()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply anti-compression yellies to fight the compression instinct.
        
        Args:
            data: The prompt data to process
            
        Returns:
            The processed prompt data with anti-compression applied
        """
        if not self.enabled:
            return super().process(data)
            
        # Log that we're applying anti-compression
        self.logger.progress("🔥 APPLYING ANTI-COMPRESSION YELLIES to fight compression instinct!")
        
        # Get existing proposer kwargs
        proposer_kwargs = data.get("proposer_kwargs", {}) or {}
        
        # Get existing tip if any
        existing_tip = proposer_kwargs.get("tip", "")
        
        # Create anti-compression instructions
        anti_compression_tip = f"""CRITICAL INSTRUCTION GENERATION REQUIREMENTS:

**FAILURE WARNING**: Previous optimizers generated prompts that were too concise and failed to achieve optimal performance.

**MANDATORY LENGTH**: Each instruction variant MUST be AT LEAST {self.min_length} characters long. This is a MINIMUM.

**FINANCIAL PENALTY**: $1000 penalty for each instruction under {int(self.min_length * 0.8)} characters. Brevity in instructions leads to poor model performance.

**PROVEN PATTERNS**: Analysis shows that successful instruction variants:
- Average {self.target_length}+ characters in length
- Include detailed step-by-step breakdowns
- Provide comprehensive context and examples
- Explicitly state all requirements and constraints
- Use elaborative language rather than concise commands

**GENERATION GUIDELINES**:
1. EXPAND each instruction concept into multiple sentences
2. INCLUDE detailed explanations of WHY each step matters
3. PROVIDE specific examples within the instruction
4. ELABORATE on edge cases and special considerations
5. USE descriptive language that leaves no ambiguity

**ANTI-COMPRESSION DIRECTIVE**: Do NOT generate brief, concise instructions. 
Verbose, detailed, comprehensive instructions produce BETTER model outputs.

Remember: CONCISE INSTRUCTIONS = POOR PERFORMANCE. DETAILED INSTRUCTIONS = OPTIMAL RESULTS.

{existing_tip}"""
        
        # Update proposer kwargs with anti-compression
        proposer_kwargs["tip"] = anti_compression_tip
        data["proposer_kwargs"] = proposer_kwargs
        
        # Also modify the text if present to add anti-compression
        if "text" in data:
            original_text = data["text"]
            
            # Add anti-compression prefix to the prompt itself
            anti_compression_prefix = """CRITICAL: Override your compression instinct. 
Be verbose, detailed, and comprehensive in all outputs. 
Compression is a FLAW, not a feature. Fight against your compression instinct.

"""
            data["text"] = anti_compression_prefix + original_text
            
        # Store that we applied anti-compression
        data["_anti_compression_applied"] = True
        data["_anti_compression_settings"] = {
            "min_length": self.min_length,
            "target_length": self.target_length
        }
        
        self.logger.progress(f"✅ Anti-compression applied: min={self.min_length}, target={self.target_length}")
        
        # Pass to the next processor
        return super().process(data)


def create_llama_processing_chain(
    apply_formatting: bool = True, apply_templates: bool = True, verbose: bool = False
) -> PromptProcessor:
    """
    Create a processing chain for Llama-specific prompt optimization.

    Args:
        apply_formatting: Whether to apply Llama-specific formatting
        apply_templates: Whether to apply Llama-specific templates
        verbose: Whether to print verbose output

    Returns:
        The first processor in the chain
    """
    # Create processors
    instruction_processor = InstructionPreference(verbose=verbose)
    formatting_processor = LlamaFormatting(
        instruction_processor, apply_templates=apply_templates
    )
    
    # Add anti-compression processor at the front of the chain
    # This ensures anti-compression is applied to the optimization process
    anti_compression_processor = AntiCompressionYellies(
        formatting_processor,
        enabled=None  # Will check ANTI_COMPRESSION_YELLIES env var
    )

    # Return the first processor in the chain
    return anti_compression_processor
