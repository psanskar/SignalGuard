import re


class RedFlagDetector:

    def __init__(self):

        self.negation_patterns = [
            r"\bno\b",
            r"\bnever\b",
            r"\bnot\b",
            r"\bwithout\b",
            r"\bdo not\b",
            r"\bdon't\b",
            r"\bdoes not\b",
            r"\bdoesn't\b",
            r"\bwill not\b",
            r"\bwon't\b",
        ]

        self.rules = {

            "upfront_payment": {
                "severity": "high",
                "patterns": [
                    r"\bregistration fee\b",
                    r"\bprocessing fee\b",
                    r"\bapplication fee\b",
                    r"\btraining fee\b",
                    r"\bpay (a |the )?(fee|deposit)\b",
                    r"\bpay\b.{0,30}\b(registration|processing|application|training)\s+fee\b",
                    r"\bdeposit (a |the )?money\b",
                    r"\bpay upfront\b",
                ],
                "message": "The job appears to request an upfront payment."
            },

            "financial_information": {
                "severity": "high",
                "patterns": [
                    r"\b(provide|send|submit|share|give|enter|upload)\b.{0,50}\b(bank account|bank details|banking information|account number)\b",

                    r"\b(bank account|bank details|banking information|account number)\b.{0,50}\b(send|provide|submit|share|give|enter|upload)\b",

                    r"\b(provide|send|submit|share|give|enter|upload)\b.{0,50}\b(credit card|debit card)\b",

                    r"\b(credit card|debit card)\b.{0,50}\b(send|provide|submit|share|give|enter|upload)\b",

                    r"\b(banking credentials|online banking credentials|banking password|banking login)\b",

                    r"\b(pin|cvv|security code)\b.{0,50}\b(bank|card|account)\b",
                ],
                "message": "The job appears to request sensitive financial information."
            },

            "sensitive_personal_information": {
                "severity": "high",
                "patterns": [
                    r"\bsocial security\b",
                    r"\bssn\b",
                    r"\bpassport number\b",
                    r"\bdriver'?s license\b",
                    r"\bgovernment id\b",
                    r"\bidentity document\b",
                ],
                "message": "The job appears to request sensitive personal information."
            },

            "cryptocurrency": {
                "severity": "high",
                "patterns": [
                    r"\bbitcoin\b",
                    r"\bbtc\b",
                    r"\bcrypto(currency)?\b",
                    r"\busdt\b",
                    r"\bethereum\b",
                    r"\beth\b",
                    r"\bcrypto wallet\b",
                    r"\bwallet address\b",
                ],
                "message": "The job contains cryptocurrency-related payment or transaction language."
            },

            "messaging_platform_recruitment": {
                "severity": "medium",
                "patterns": [
                    r"\b(contact|message|reach|connect)\s+(me|us|the recruiter|our recruiter|the hiring manager)\s+(on|via)\s+(telegram|whatsapp)\b",
                ],
                "message": "Recruitment appears to rely on external messaging platforms."
            },

            "urgency_pressure": {
                "severity": "medium",
                "patterns": [
                    r"\bact now\b",
                    r"\bapply immediately\b",
                    r"\bimmediate(ly)?\b",
                    r"\burgent\b",
                    r"\blimited (time|slots)\b",
                    r"\bonly today\b",
                    r"\bavailable for a limited time\b",
                ],
                "message": "The job uses urgency or pressure-based language."
            },

            "unrealistic_income": {
                "severity": "medium",
                "patterns": [

                    # Guaranteed income / earnings / salary / pay
                    r"\bguaranteed\s+(income|earnings?|salary|pay)\b",

                    # Easy / unrealistic earning claims
                    r"\beasy\s+money\b",
                    r"\bget\s+rich\b",

                    # earn/make a numeric amount with per/a day/week/month/year
                    r"\b(?:earn|make)\s+[₹$]?\d[\d,.]*\s*(?:per|a)\s*(?:day|week|month|year)\b",

                    # earn/make a numeric amount with daily/weekly/monthly/yearly
                    r"\b(?:earn|make)\s+[₹$]?\d[\d,.]*\s*(?:daily|weekly|monthly|yearly)\b",

                    # guaranteed BEFORE the amount
                    r"\bguaranteed\s+[₹$]?\d[\d,.]*\s*(?:per|a)\s*(?:day|week|month|year)\b",
                    r"\bguaranteed\s+[₹$]?\d[\d,.]*\s*(?:daily|weekly|monthly|yearly)\b",

                    # guaranteed AFTER the amount
                    # Example: "$4,500 weekly guaranteed"
                    r"[₹$]?\d[\d,.]*\s*(?:per|a)\s*(?:day|week|month|year)\s+guaranteed\b",
                    r"[₹$]?\d[\d,.]*\s*(?:daily|weekly|monthly|yearly)\s+guaranteed\b",

                    # "guaranteed $500/week" style
                    r"\bguaranteed\s+[₹$]?\d[\d,.]*\s*/\s*(?:day|week|month|year)\b",

                    # "earn up to $5000 weekly" / "make up to $5000 monthly"
                    r"\b(?:earn|make)\s+(?:up\s+to\s+)?[₹$]?\d[\d,.]*\s*(?:daily|weekly|monthly|yearly)\b",

                    # Numeric income followed by "per week" etc.
                    r"\b[₹$]?\d[\d,.]*\s*(?:per|a)\s*(?:day|week|month|year)\s+(?:income|earnings?|salary|pay)\b",
                ],
                "message": "The job contains potentially unrealistic income claims."
            },

            "fake_check": {
                "severity": "high",
                "patterns": [
                    r"\bfake check\b",
                    r"\bcash (the )?check\b",
                    r"\bdeposit (the )?check\b",
                    r"\bdeposit (the )?cheque\b",
                    r"\bmobile deposit\b",
                    r"\boverpayment\b",
                ],
                "message": "The job contains language associated with check or overpayment scams."
            },
        }


    def _is_negated(self, text, match_start):
        """
        Check whether a matched phrase is preceded by
        a negation within a small surrounding context window.
        """

        context_start = max(0, match_start - 80)
        context = text[context_start:match_start]

        for pattern in self.negation_patterns:
            if re.search(pattern, context):
                return True

        return False


    def detect(self, text):

        if not text:
            return []

        text = str(text).lower()

        detected = []

        for category, rule in self.rules.items():

            matches = []

            for pattern in rule["patterns"]:

                for match in re.finditer(pattern, text):

                    # Ignore phrases that appear to be explicitly negated
                    if self._is_negated(text, match.start()):
                        continue

                    matches.append(match.group())

            if matches:

                # Remove duplicate matches
                matches = list(dict.fromkeys(matches))

                detected.append({
                    "category": category,
                    "severity": rule["severity"],
                    "message": rule["message"],
                    "matches": matches,
                    "match_count": len(matches)
                })

        return detected