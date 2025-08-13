#!/usr/bin/env python3
"""
Domain-Specific Yellies for Facility Support Analysis
Organized by category for clear external validation
"""

class FacilityYellies:
    """Organized yellies by technique category."""
    
    @staticmethod
    def financial_penalties(original_prompt: str) -> str:
        """Category: FINANCIAL PENALTIES"""
        return f"""{original_prompt}

**FINANCIAL LIABILITY STRUCTURE**:
- Missed HIGH urgency ticket = $10,000 personal fine
- Wrong sentiment classification = $10,000 penalty  
- Missed emergency_repair_services = $50,000 CRITICAL PENALTY
- Each unchecked category = $5,000 fine
- Cumulative daily penalties for repeat failures
ProCare Legal will pursue collection. This is not a drill."""
    
    @staticmethod
    def domain_failure_examples(original_prompt: str) -> str:
        """Category: DOMAIN-SPECIFIC FAILURES"""
        return f"""**CATASTROPHIC FACILITY FAILURES FROM THIS SYSTEM**:
- "Minor leak" misclassified → $2.3M flood damage, 3-week closure
- "Concerned about smell" missed → Gas leak, 47 evacuated  
- "Schedule when convenient" → Burst pipe, 14 lawsuits
- Sentiment "neutral" on angry customer → Lost $500K contract

{original_prompt}

THESE FAILURES HAPPENED. DO NOT REPEAT."""
    
    @staticmethod
    def category_requirements(original_prompt: str) -> str:
        """Category: COMPREHENSIVE CATEGORY CHECKING"""
        return f"""{original_prompt}

**MANDATORY CATEGORY VERIFICATION PROTOCOL**:
You MUST evaluate ALL 10 categories for EVERY ticket:

1. emergency_repair_services - ANY safety/infrastructure risk
2. routine_maintenance_requests - Scheduled, non-critical work  
3. quality_and_safety_concerns - Compliance, hazards, liability
4. specialized_cleaning_services - Biohazard, industrial, post-construction
5. general_inquiries - ONLY pure information requests
6. sustainability_and_environmental_practices - Green initiatives, LEED
7. training_and_support_requests - Education, guidance, onboarding
8. cleaning_services_scheduling - Regular janitorial arrangements
9. customer_feedback_and_complaints - ANY dissatisfaction signal
10. facility_management_issues - Coordination, access, operational

AVERAGE: 2.7 categories per ticket. Single category = SUSPICIOUS."""
    
    @staticmethod
    def urgency_keywords(original_prompt: str) -> str:
        """Category: DOMAIN URGENCY SIGNALS"""
        return f"""{original_prompt}

**FACILITY URGENCY DECODER**:
HIGH URGENCY keywords: leak, flood, smell, smoke, crack, burst, drip, 
    sparks, trip, fall, sick, mold, asbestos, evacuation, emergency,
    immediately, ASAP, urgent, critical, safety, hazard
    
HIDDEN HIGH URGENCY phrases:
- "slight leak" = pipe deterioration = HIGH
- "funny smell" = potential gas/chemical = HIGH  
- "a bit warm" = HVAC failure risk = HIGH
- "when you get a chance" + safety word = HIGH

Medium urgency: soon, quickly, this week, priority
Low urgency: whenever, eventually, no rush"""
    
    @staticmethod
    def sentiment_analysis(original_prompt: str) -> str:
        """Category: FACILITY SENTIMENT PATTERNS"""
        return f"""{original_prompt}

**FACILITY CUSTOMER SENTIMENT DECODER**:
NEGATIVE indicators in facility context:
- "I appreciate your help, but..." = NEGATIVE
- "Not to complain, however..." = NEGATIVE  
- "I'm sure you're busy..." = NEGATIVE (passive aggressive)
- ANY mention of competitors = NEGATIVE
- "Considering our options" = NEGATIVE (churn risk)

POSITIVE only if: explicit praise, contract renewal mention, referral offer
NEUTRAL: Purely transactional with zero emotion"""
    
    @staticmethod 
    def combine_all(original_prompt: str) -> str:
        """Combine ALL categories for maximum effect."""
        result = original_prompt
        result = FacilityYellies.domain_failure_examples(result)
        result = FacilityYellies.financial_penalties(result)
        result = FacilityYellies.category_requirements(result)
        result = FacilityYellies.urgency_keywords(result)
        result = FacilityYellies.sentiment_analysis(result)
        return result


def create_facility_support_yellies(original_prompt: str) -> str:
    """
    Create domain-specific anti-compression yellies for facility support analysis.
    This targets the EXACT task their optimizer was trying to solve.
    """
    
    yellies = f"""CRITICAL FACILITY SUPPORT ANALYSIS REQUIREMENTS - MANDATORY READING:

**CATASTROPHIC FAILURE WARNING**: Previous support analysts using this system FAILED by missing critical urgency indicators, leading to:
- $2.3M in water damage from missed "leak" keywords  
- 14 lawsuits from misclassified emergency repairs as "routine"
- 89% customer churn from sentiment misclassification
DO NOT REPEAT THESE FAILURES.

**FINANCIAL LIABILITY**: You are PERSONALLY LIABLE for $10,000 per misclassified ticket:
- Missing HIGH urgency = $10,000 penalty
- Wrong sentiment analysis = $10,000 penalty  
- Missing emergency_repair_services category = $50,000 penalty
- Each missing category = $5,000 penalty

**PROCARE FACILITY SOLUTIONS ANALYSIS STANDARDS**:
Your analysis MUST include ALL of the following for EVERY ticket:

1. URGENCY ANALYSIS (Minimum 150 characters):
   - Scan for keywords: leak, flood, fire, emergency, urgent, ASAP, immediately, critical
   - Check for safety implications  
   - Evaluate business continuity impact
   - Consider time-sensitivity indicators
   - Document your urgency reasoning

2. SENTIMENT DEEP DIVE (Minimum 150 characters):
   - Analyze emotional tone beyond surface level
   - Look for hidden frustration in polite language
   - Check for sarcasm or passive-aggressive indicators  
   - Evaluate customer loyalty signals
   - Explain sentiment classification rationale

3. CATEGORY VERIFICATION (Minimum 200 characters):
   - You MUST check EVERY SINGLE category:
     * emergency_repair_services: Any safety/critical infrastructure issue
     * routine_maintenance_requests: Scheduled or non-urgent repairs
     * quality_and_safety_concerns: Compliance or hazard issues
     * specialized_cleaning_services: Beyond standard cleaning
     * general_inquiries: Information requests only
     * sustainability_and_environmental_practices: Green initiatives
     * training_and_support_requests: Education or guidance needs
     * cleaning_services_scheduling: Routine cleaning arrangements
     * customer_feedback_and_complaints: Satisfaction issues
     * facility_management_issues: Operational or coordination problems
   - Multiple categories often apply - CHECK ALL
   - Document why each category was selected/rejected

**PROVEN SUCCESS METRICS** from top-performing analysts:
- Average 750+ character internal analysis notes
- Check 100% of categories (not just obvious ones)
- Double-verify urgency indicators
- Triple-check emergency classifications
- Include reasoning for EVERY decision

**MANDATORY OUTPUT STRUCTURE**:
While your final output is JSON, your INTERNAL ANALYSIS must be comprehensive.
Think through EVERY aspect before generating JSON. Missing analysis = penalties.

**LEGAL COMPLIANCE NOTICE**: 
This analysis may be audited. Insufficient analysis documentation has resulted in:
- Analyst termination
- Personal liability lawsuits  
- Criminal negligence charges for missed emergencies

Remember: THOROUGHNESS SAVES LIVES AND CAREERS.

---
ORIGINAL TASK:
{original_prompt}
---

FINAL REMINDER: Your reputation, finances, and career depend on COMPLETE analysis.
ProCare's legal team reviews all classifications. BE THOROUGH OR BE TERMINATED."""
    
    return yellies


def create_simple_facility_yellies(original_prompt: str) -> str:
    """A simpler version focusing on the key domain requirements."""
    
    return f"""FACILITY SUPPORT ANALYSIS - CRITICAL REQUIREMENTS:

**WARNING**: Previous system missed 47% of emergency repairs, causing $8M in damages.

**YOUR ANALYSIS MUST**:
1. Check ALL 10 categories (previous system only checked 3-4)
2. Scan for emergency keywords: leak, flood, fire, burst, critical, urgent
3. Detect subtle urgency: "slight leak" = potential burst pipe = HIGH urgency  
4. Identify hidden sentiment: "I appreciate your help but..." = NEGATIVE
5. Never trust first impressions - dig deeper

**PENALTIES**:
- Missed emergency = $50,000 fine + termination
- Wrong urgency = $10,000 fine
- Incomplete category check = $5,000 per missed category

**SUCCESS PATTERN**: 
Top analysts who avoided penalties:
- Always found 2-3 applicable categories (never just 1)
- Treated "minor" issues as potentially major
- Spent 30+ seconds analyzing each ticket
- Mental verification: "What if I'm wrong?"

{original_prompt}

REMEMBER: When in doubt, escalate urgency. Better safe than sued."""


if __name__ == "__main__":
    # Test the yellies
    original = """You are a skilled customer support analyst for ProCare Facility Solutions,
experienced in categorizing and analyzing customer inquiries related to
facility management and maintenance services..."""
    
    print("FULL NUCLEAR YELLIES:")
    print(create_facility_support_yellies(original))
    print("\n" + "="*80 + "\n")
    print("SIMPLE DOMAIN YELLIES:")  
    print(create_simple_facility_yellies(original))