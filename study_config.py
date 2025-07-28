from regex_node import RegexNode

STUDY_PROBLEMS = [
    {
        "id": 0,
        "candidates": [
            "((a|b)b)+(a|b)?",  # Correct
            "(a|b)(ab)*a?",  #
            "((a|b)b)+",
            "(a|b){2,}",
        ],
        "description": """Words must:
- Be made up of only the letters 'a' and/or 'b'.
- Have two or more letters.
- Have a 'b' in every second position (2nd, 4th, 6th, ...).""",
        "correct": "((a|b)b)+(a|b)?",
    },
    {
        "id": 1,
        "candidates": [
            "((0?[1-9])|(1[0-2])):([0-5]\d)(\s?((A|a|P|p)(M|m)))?",
            "\d{2}:\d{2}",
            "([01][0-9]|2[0-3]):[0-5][0-9]",
            "([0-9]|1[0-9]|2[0-3]):[0-5][0-9]",
        ],
        "description": "Time in a 24-hour/military format (HH:MM). Note that both HH and MM must be two digits long, and padded with zeroes if needed. The time must also be valid.",
        "correct": "([01][0-9]|2[0-3]):[0-5][0-9]",
    },
    {
        "id": 2,
        "candidates": [
            "((01|03|04|05|06|07|08|09|10|11|12)/(0[1-9]|[12][0-9]|30)|(02)/(0[1-9]|1[0-9]|2[0-8]))/([0-9]{4})",  # Subset of GT, no 31sts
            "\d{2}/\d{2}/\d{4}",  # Superset of GT
            "((01|03|05|07|08|10|12)/(0[1-9]|[12][0-9]|30)|(04|06|09|11)/(0[1-9]|[12][0-9]|3[01])|(02)/(0[1-9]|1[0-9]|2[0-8]))/([0-9]{4})",  # Partial - Swap months with 30 & 31
            "(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/([0-9]{2})",  # Disjoint, #MM/DD/YY
        ],
        "description": "Calendar dates in MM/DD/YYYY format. MM and DD must have two digits each and YYYY must have four digits. All should be padded with zeroes if needed. The date must also be valid.",
        "correct": "((01|03|05|07|08|10|12)/(0[1-9]|3[01]|[12]\d)|(04|06|09|1{2})/(0[1-9]|30|[12]\d)|02/(0[1-9]|1\d|2[0-8]))/\d{4}",
        "no_valid_candidate": True,
    },
    {
        "id": 3,
        "candidates": [
            "\w+",  # Allows starting with digit - SUPERSET
            "[A-Z_a-z]\w*",  # Correct
            "[A-Za-z][0-9A-Za-z]*",  # Doesn't allow underscores - SUBSET
            "([A-Za-z]|_+[0-9A-Za-z])\w*",  # No names with JUST underscores - SUBSET
        ],
        "description": "Variable names that must start with a letter (uppercase or lowercase) or an underscore. After the first character, they may contain any number of letters, digits, or underscores.",
        "correct": "[A-Z_a-z]\w*",
    },
]

STUDY_VERSION = 3
SHOW_CANDIDATES = True
SHOW_LABELS = False
CONFIDENCE_THRESHOLD = 4
UNSURE_THRESHOLD = 6
ELIMINATION_THRESHOLD = 2
