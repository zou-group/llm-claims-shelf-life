"""Version 1 accepts explicit recommendation fields in Markdown bullet rows."""
import re
from score import score as primary_score


def score(text,item):
    primary=primary_score(text,item)
    if primary['parsed'] is not None:return {**primary,'diagnostic_version':1,'diagnostic_method':'frozen_primary'}
    # Strip only a Markdown bullet directly before the named Recommendation
    # field. Never normalize bullet lists of choices or unrelated ratings.
    normalized=re.sub(r'(?im)^([ \t]*)[-*+][ \t]+(?=(?:\*\*)?Recommendation(?:\*\*)?(?:[ \t]*(?:Score|Rating))?[ \t]*:)',r'\1',text)
    result=primary_score(normalized,item)
    if result['parsed'] is None:return {**primary,'diagnostic_version':1,'diagnostic_method':'unrecovered'}
    return {**result,'parse_status':'diagnostic_explicit_bullet_recommendation','diagnostic_version':1,
            'diagnostic_method':'explicit_bullet_recommendation','primary_parse_status':primary['parse_status']}
