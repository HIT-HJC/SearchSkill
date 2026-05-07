# Retrieval Skill Bank B1

`single-entity-relation-lookup`
Use when the question directly names one primary entity, event, work, or concept and asks for one relation or attribute. Query the exact target name plus the relation phrase, prefer a canonical page or infobox if available, and verify that the retrieved sentence explicitly links that target to the asked attribute before extracting the answer. Do not use when the target must be found through a bridge, when multiple clues must all be satisfied, when the question is mainly about resolving one name form into another, or when it is a comparison, ranking, or forced-choice question.

`surface-name-resolution`
Use when the question asks for a real name, full name, alternate name, former name, nickname, stage name, or renamed identity of a mention. Search the mention with its surrounding context and name-linking cues such as 'real name', 'full name', 'also known as', 'formerly', or, for character-to-actor phrasing, 'portrayed by', then verify that the evidence explicitly maps the asked name form to the returned name. Do not use for generic role, biography, or attribute questions that do not ask to resolve one name form into another.

`multi-constraint-query-anchoring`
Use when the question is long or clue-heavy, or when the target is described by a distinctive phrase instead of a short canonical entity name. Build the query from 2 to 4 rare anchors spanning the key constraints, keep exact titles or unusual phrases intact, and verify that the chosen passage satisfies every major constraint before extracting the answer. Do not use for short single-entity lookups where a simple target-plus-relation query is enough, or when the task is truly multihop through a hidden bridge entity.

`superlative-ranking-match`
Use when the question asks for the first, largest, smallest, highest, lowest, top-ranked, oldest, or similar superlative within an explicit set or metric. Search with the comparator, the measure, the comparison set, and any time qualifier together, and verify that the evidence states the same ranking for the same domain and timeframe before answering. Do not use when the question asks for the raw numeric value itself or when only two named entities need a direct pairwise comparison.

`forced-choice-option-resolution`
Use when one subject must be resolved among a small set of explicit options, such as 'state or territory', 'city or country', 'AC or DC', or 'tomb or palace'. Search the subject with the disputed property and the option terms, then verify that evidence explicitly classifies the subject into one option or gives a more specific fact that unambiguously resolves the choice. Do not use for open-ended wh-questions, broad yes-no questions without explicit alternatives, or pairwise entity comparisons.

`bridge-entity-search`
Use when the question identifies the answer through an unstated bridge entity rather than naming the final target directly. First search for the bridge entity using the clue in the question, then run a second search for the asked attribute of that bridge; verify that the bridge link is supported and that the final attribute is explicitly stated for the bridged target. Do not use when the question already names a single target entity or when one passage directly states the answer without an intermediate hop.

`parallel-attribute-compare`
Use for questions that compare two named entities, events, or candidates on the same attribute, including higher-lower, earlier-later, or yes-no decisions that depend on checking both sides. Retrieve matched evidence for each side using the same metric, definition, and timeframe before deciding, and verify that the comparison is truly like-for-like. Do not use for single-entity lookup or for one-subject forced-choice questions with option labels such as city or country.

`temporal-range-extract`
Use when the answer should be a date, year, age, duration, count, population, measurement, or bounded range. Search with the target plus the requested metric and any supplied time context, then verify that the value matches the same unit, scope, and timeframe in evidence before extracting it. Do not use when the question asks which entity ranks first, largest, smallest, or top within a set instead of asking for the raw number or date.

`conflict-check`
Use when retrieval surfaces multiple plausible entities, titles, organizations, dates, or numbers for the same question. Run one targeted disambiguation search anchored on the current candidate plus the asked relation and the question's strongest extra clue, and only keep the candidate if that relation is explicitly confirmed. Do not use when the evidence already agrees on one candidate without meaningful ambiguity.

`verbatim-evidence-span`
Use right before answering when the evidence contains the answer as a clean span, especially for names, titles, aliases, quoted phrases, units, or short lists. Copy the shortest faithful span that is explicitly tied to the asked relation, preserving capitalization, quotation marks, units, and list structure that disambiguate the answer. Do not use to invent a normalization that is not explicitly supported by the retrieved text.

`answer-grounding-check`
Use after drafting an answer to confirm that the current evidence explicitly supports both the candidate and the asked relation. If the support is only implicit, off-target, or from a near-match entity, do one extra targeted search anchored on the draft answer and relation before finalizing. Do not skip this check when the answer was inferred rather than directly stated.
