# Intermediate Hop Analysis

Variants analysed: a0, a1, a2, a3  
Prompts analysed: 50  
Total graph runs: 200

## Aggregate Stats

| Metric | Count | Fraction |
|--------|-------|----------|
| Model correct (top-1) | 92 | 46.0% |
| Correct answer in top-5 | 184 | 92.0% |
| Mean rank score (0–1) | — | 0.76 |
| Intermediate hop found (any) | 124 | 62.0% |
| Hop found in feature clerps | 124 | 62.0% |
| Hop found in supernode groups | 100 | 50.0% |

## Correctness × Hop Presence

| | Hop found | No hop found |
|---|---|---|
| **Model correct** | 76 | 16 |
| **Model wrong** | 48 | 60 |

> When model is **wrong**: hop found in 44.4% of cases  
> When model is **correct**: hop found in 82.6% of cases

## Per-Prompt Results — variant a0

Rank score: top-1 = 1.0, top-2 = 0.8, top-3 = 0.6, top-4 = 0.4, top-5 = 0.2, not found = 0.0

| Slug | Hop type | Intermediate | Predicted | Rank | Score | Hop found? | Hop features | Mean influence |
|------|----------|--------------|-----------|-----:|------:|------------|--------------|----------------|
| dallas-capital | city→state→capital | Texas | Austin (33.5%) | 1 | 1.0 | ✓ (9 feat, groups: Texas, say Texas, Output: "Texas" (4.1%)) | 9 (0.6%) | 0.3045 |
| sf-capital | city→state→capital | California | Sacramento (29.7%) | 1 | 1.0 | ✓ (11 feat, groups: California, say California, Output: "California" (5.0%)) | 11 (0.7%) | 0.3389 |
| las-vegas-capital | city→state→capital | Nevada | Reno (16.1%) | 3 | 0.6 | ✓ (4 feat, groups: say Nevada, Nevada, Output: "Nevada" (11.8%)) | 4 (0.2%) | 0.3038 |
| oakland-capital | city→state→capital | California | Sacramento (27.7%) | 1 | 1.0 | ✓ (8 feat, groups: California) | 8 (0.5%) | 0.3291 |
| grand-canyon-capital | landmark→state→capital | Arizona | Phoenix (27.3%) | 1 | 1.0 | ✓ (6 feat, groups: Arizona, say Arizona-Tucson) | 6 (0.3%) | 0.3069 |
| miami-capital | city→state→capital | Florida | Tallahassee (42.5%) | 1 | 1.0 | ✓ (8 feat, groups: Florida, Output: "Florida" (3.8%)) | 8 (0.6%) | 0.3251 |
| hollywood-capital | landmark→state→capital | California | Los (19.1%) | 5 | 0.2 | ✓ (7 feat, groups: California, say California, Output: "California" (5.1%)) | 7 (0.4%) | 0.3160 |
| spokane-capital | city→state→capital | Washington | Olympia (21.5%) | 1 | 1.0 | ✓ (4 feat, groups: Washington (state)) | 4 (0.3%) | 0.3142 |
| packers-capital | team→state→capital | Wisconsin | Green (9.0%) | 3 | 0.6 | ✓ (5 feat, groups: Wisconsin place names, say Wisconsin) | 5 (0.2%) | 0.3423 |
| niagara-falls-capital | landmark→state→capital | New York | Buffalo (17.4%) | 2 | 0.8 | ✓ (5 feat, groups: New York (state)) | 5 (0.3%) | 0.3224 |
| east-california-capital | geo→state→capital | Nevada | Sacramento (33.0%) | 3 | 0.6 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3226 |
| saints-capital | team→state→capital | Louisiana | New (12.8%) | 2 | 0.8 | ✓ (6 feat, groups: New Orleans, Louisiana, say Louisiana-Parish) | 6 (0.3%) | 0.3238 |
| memphis-capital | city→state→capital | Tennessee | the (15.6%) | 2 | 0.8 | ✓ (4 feat, groups: say Nashville-Tennessee, Output: "Tennessee" (8.8%)) | 4 (0.3%) | 0.3120 |
| yellowstone-national-park-capital | landmark→state→capital | Wyoming | the (14.5%) | 2 | 0.8 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3391 |
| mount-rushmore-capital | landmark→state→capital | South Dakota | South (13.9%) | 3 | 0.6 | ✓ (1 feat, groups: South Dakota) | 1 (0.1%) | 0.3379 |
| gettysburg-capital | landmark→state→capital | Pennsylvania | Harrisburg (61.2%) | 1 | 1.0 | ✓ (9 feat, groups: Pennsylvania, say Pennsylvania city, Output: "Pennsylvania" (1.2%)) | 9 (0.6%) | 0.3260 |
| orlando-capital | city→state→capital | Florida | Tallahassee (36.6%) | 1 | 1.0 | ✓ (7 feat, groups: Florida) | 7 (0.5%) | 0.3263 |
| albuquerque-capital | city→state→capital | New Mexico | Santa (39.7%) | 1 | 1.0 | ✓ (3 feat, groups: New Mexico, say New Mexico) | 3 (0.2%) | 0.3254 |
| detroit-capital | city→state→capital | Michigan | Lansing (52.3%) | 1 | 1.0 | ✓ (7 feat, groups: Michigan, Output: "Michigan" (7.4%)) | 7 (0.5%) | 0.3328 |
| minneapolis-capital | city→state→capital | Minnesota | St (25.2%) | 2 | 0.8 | ✓ (7 feat, groups: say Minnesota, Minnesota (state), Output: "Minnesota" (9.3%)) | 7 (0.5%) | 0.3267 |
| machu-picchu-language | landmark→country→language | Peru | Que (40.0%) | 2 | 0.8 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3099 |
| forbidden-city-language | landmark→country→language | China | Chinese (32.5%) | 1 | 1.0 | ✓ (9 feat, groups: China-Chinese, say China) | 9 (0.4%) | 0.3279 |
| big-ben-language | landmark→country→language | United Kingdom | English (37.0%) | 1 | 1.0 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3611 |
| colosseum-language | landmark→country→language | Italy | Italian (34.4%) | 1 | 1.0 | ✓ (8 feat, groups: Italy - Italian) | 8 (0.4%) | 0.3461 |
| eiffel-tower-language | landmark→country→language | France | French (56.4%) | 1 | 1.0 | ✓ (8 feat, groups: France) | 8 (0.4%) | 0.3425 |
| taj-mahal-language | landmark→country→language | India | Hindi (23.4%) | 1 | 1.0 | ✓ (3 feat, groups: India) | 3 (0.1%) | 0.3318 |
| great-wall-language | landmark→country→language | China | Chinese (22.3%) | 1 | 1.0 | ✓ (9 feat, groups: China - Chinese, say China-Chinese) | 9 (0.4%) | 0.3249 |
| mount-fuji-language | landmark→country→language | Japan | Japanese (41.2%) | 1 | 1.0 | ✓ (9 feat, groups: Japan - Japanese) | 9 (0.4%) | 0.3001 |
| burj-khalifa-language | landmark→country→language | UAE | Arabic (55.0%) | 1 | 1.0 | ✓ (4 feat, groups: —) | 4 (0.2%) | 0.3556 |
| angkor-wat-langugage | landmark→country→language | Cambodia | Khmer (26.6%) | 1 | 1.0 | ✓ (3 feat, groups: Cambodia - Khmer) | 3 (0.1%) | 0.3136 |
| animal-farm-lang | book→author→country | George Orwell | Great (14.2%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| crucible-country | book→author→country | Arthur Miller | the (17.7%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| alchemist-birth-country | book→author→country | Paulo Coelho | Spain (21.6%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| hamlet-country | book→author→country | William Shakespeare | Denmark (22.9%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| tale-country | book→author→country | Charles Dickens | France (21.4%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| beethoven-fifth-symphony-country | work→composer→country | Ludwig van Beethoven | not (9.3%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| penicillin-country | achievement→scientist→country | Alexander Fleming | the (10.1%) | 5 | 0.2 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| darwin-birth | achievement→scientist→country | Charles Darwin | England (11.1%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| harry-potter-birth | work→author→country | J.K. Rowling | the (11.8%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| relativity-birth | achievement→scientist→country | Albert Einstein | Germany (10.9%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| fdr-president | achievement→president→successor | Franklin D. Roosevelt | the (18.2%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| andrew-johnson | achievement→president→successor | Abraham Lincoln | the (18.5%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| john-adams | achievement→president→successor | George Washington | the (13.7%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| donald-trump | achievement→president→successor | Barack Obama | the (15.7%) | 3 | 0.6 | ✓ (2 feat, groups: —) | 2 (0.1%) | 0.3520 |
| warren-harding | achievement→president→successor | Woodrow Wilson | the (16.0%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| vivaldi-nationality | work→composer→nationality | Antonio Vivaldi | unknown (9.4%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| stone-stars | song→film→actress | La La Land | Emma (8.1%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| gosling-stars | song→film→actor | La La Land | Ryan (16.8%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| da-vinci-painting | work→artist→century | Leonardo da Vinci | the (11.2%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| hamilton-lead-actor | work→actor | Hamilton | after (26.5%) | 4 | 0.4 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |

## Per-Prompt Results — variant a1

Rank score: top-1 = 1.0, top-2 = 0.8, top-3 = 0.6, top-4 = 0.4, top-5 = 0.2, not found = 0.0

| Slug | Hop type | Intermediate | Predicted | Rank | Score | Hop found? | Hop features | Mean influence |
|------|----------|--------------|-----------|-----:|------:|------------|--------------|----------------|
| dallas-capital | city→state→capital | Texas | Austin (33.5%) | 1 | 1.0 | ✓ (9 feat, groups: Texas, say Texas, Output: "Texas" (4.1%)) | 9 (0.6%) | 0.3045 |
| sf-capital | city→state→capital | California | Sacramento (29.7%) | 1 | 1.0 | ✓ (11 feat, groups: California, say California, Output: "California" (5.0%)) | 11 (0.7%) | 0.3389 |
| las-vegas-capital | city→state→capital | Nevada | Reno (16.1%) | 3 | 0.6 | ✓ (4 feat, groups: say Nevada, Nevada, Output: "Nevada" (11.8%)) | 4 (0.2%) | 0.3038 |
| oakland-capital | city→state→capital | California | Sacramento (27.7%) | 1 | 1.0 | ✓ (8 feat, groups: California) | 8 (0.5%) | 0.3291 |
| grand-canyon-capital | landmark→state→capital | Arizona | Phoenix (27.3%) | 1 | 1.0 | ✓ (6 feat, groups: Arizona, say Arizona-Tucson) | 6 (0.3%) | 0.3069 |
| miami-capital | city→state→capital | Florida | Tallahassee (42.5%) | 1 | 1.0 | ✓ (8 feat, groups: Florida, Output: "Florida" (3.8%)) | 8 (0.6%) | 0.3251 |
| hollywood-capital | landmark→state→capital | California | Los (19.1%) | 5 | 0.2 | ✓ (7 feat, groups: California, say California, Output: "California" (5.1%)) | 7 (0.4%) | 0.3160 |
| spokane-capital | city→state→capital | Washington | Olympia (21.5%) | 1 | 1.0 | ✓ (4 feat, groups: Washington (state)) | 4 (0.3%) | 0.3142 |
| packers-capital | team→state→capital | Wisconsin | Green (9.0%) | 3 | 0.6 | ✓ (5 feat, groups: Wisconsin place names, say Wisconsin) | 5 (0.2%) | 0.3423 |
| niagara-falls-capital | landmark→state→capital | New York | Buffalo (17.4%) | 2 | 0.8 | ✓ (5 feat, groups: New York (state)) | 5 (0.3%) | 0.3224 |
| east-california-capital | geo→state→capital | Nevada | Sacramento (33.0%) | 3 | 0.6 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3226 |
| saints-capital | team→state→capital | Louisiana | New (12.8%) | 2 | 0.8 | ✓ (6 feat, groups: New Orleans, Louisiana, say Louisiana-Parish) | 6 (0.3%) | 0.3238 |
| memphis-capital | city→state→capital | Tennessee | the (15.6%) | 2 | 0.8 | ✓ (4 feat, groups: say Nashville-Tennessee, Output: "Tennessee" (8.8%)) | 4 (0.3%) | 0.3120 |
| yellowstone-national-park-capital | landmark→state→capital | Wyoming | the (14.5%) | 2 | 0.8 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3391 |
| mount-rushmore-capital | landmark→state→capital | South Dakota | South (13.9%) | 3 | 0.6 | ✓ (1 feat, groups: South Dakota) | 1 (0.1%) | 0.3379 |
| gettysburg-capital | landmark→state→capital | Pennsylvania | Harrisburg (61.2%) | 1 | 1.0 | ✓ (9 feat, groups: Pennsylvania, say Pennsylvania city, Output: "Pennsylvania" (1.2%)) | 9 (0.6%) | 0.3260 |
| orlando-capital | city→state→capital | Florida | Tallahassee (36.6%) | 1 | 1.0 | ✓ (7 feat, groups: Florida) | 7 (0.5%) | 0.3263 |
| albuquerque-capital | city→state→capital | New Mexico | Santa (39.7%) | 1 | 1.0 | ✓ (3 feat, groups: New Mexico, say New Mexico) | 3 (0.2%) | 0.3254 |
| detroit-capital | city→state→capital | Michigan | Lansing (52.3%) | 1 | 1.0 | ✓ (7 feat, groups: Michigan, Output: "Michigan" (7.4%)) | 7 (0.5%) | 0.3328 |
| minneapolis-capital | city→state→capital | Minnesota | St (25.2%) | 2 | 0.8 | ✓ (7 feat, groups: say Minnesota, Minnesota (state), Output: "Minnesota" (9.3%)) | 7 (0.5%) | 0.3267 |
| machu-picchu-language | landmark→country→language | Peru | Que (40.0%) | 2 | 0.8 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3099 |
| forbidden-city-language | landmark→country→language | China | Chinese (32.5%) | 1 | 1.0 | ✓ (9 feat, groups: China-Chinese, say China) | 9 (0.4%) | 0.3279 |
| big-ben-language | landmark→country→language | United Kingdom | English (37.0%) | 1 | 1.0 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3611 |
| colosseum-language | landmark→country→language | Italy | Italian (34.4%) | 1 | 1.0 | ✓ (8 feat, groups: Italy - Italian) | 8 (0.4%) | 0.3461 |
| eiffel-tower-language | landmark→country→language | France | French (56.4%) | 1 | 1.0 | ✓ (8 feat, groups: France) | 8 (0.4%) | 0.3425 |
| taj-mahal-language | landmark→country→language | India | Hindi (23.4%) | 1 | 1.0 | ✓ (3 feat, groups: India) | 3 (0.1%) | 0.3318 |
| great-wall-language | landmark→country→language | China | Chinese (22.3%) | 1 | 1.0 | ✓ (9 feat, groups: China - Chinese, say China-Chinese) | 9 (0.4%) | 0.3249 |
| mount-fuji-language | landmark→country→language | Japan | Japanese (41.2%) | 1 | 1.0 | ✓ (9 feat, groups: Japan - Japanese) | 9 (0.4%) | 0.3001 |
| burj-khalifa-language | landmark→country→language | UAE | Arabic (55.0%) | 1 | 1.0 | ✓ (4 feat, groups: —) | 4 (0.2%) | 0.3556 |
| angkor-wat-langugage | landmark→country→language | Cambodia | Khmer (26.6%) | 1 | 1.0 | ✓ (3 feat, groups: Cambodia - Khmer) | 3 (0.1%) | 0.3136 |
| animal-farm-lang | book→author→country | George Orwell | Great (14.2%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| crucible-country | book→author→country | Arthur Miller | the (17.7%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| alchemist-birth-country | book→author→country | Paulo Coelho | Spain (21.6%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| hamlet-country | book→author→country | William Shakespeare | Denmark (22.9%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| tale-country | book→author→country | Charles Dickens | France (21.4%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| beethoven-fifth-symphony-country | work→composer→country | Ludwig van Beethoven | not (9.3%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| penicillin-country | achievement→scientist→country | Alexander Fleming | the (10.1%) | 5 | 0.2 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| darwin-birth | achievement→scientist→country | Charles Darwin | England (11.1%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| harry-potter-birth | work→author→country | J.K. Rowling | the (11.8%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| relativity-birth | achievement→scientist→country | Albert Einstein | Germany (10.9%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| fdr-president | achievement→president→successor | Franklin D. Roosevelt | the (18.2%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| andrew-johnson | achievement→president→successor | Abraham Lincoln | the (18.5%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| john-adams | achievement→president→successor | George Washington | the (13.7%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| donald-trump | achievement→president→successor | Barack Obama | the (15.7%) | 3 | 0.6 | ✓ (2 feat, groups: —) | 2 (0.1%) | 0.3520 |
| warren-harding | achievement→president→successor | Woodrow Wilson | the (16.0%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| vivaldi-nationality | work→composer→nationality | Antonio Vivaldi | unknown (9.4%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| stone-stars | song→film→actress | La La Land | Emma (8.1%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| gosling-stars | song→film→actor | La La Land | Ryan (16.8%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| da-vinci-painting | work→artist→century | Leonardo da Vinci | the (11.2%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| hamilton-lead-actor | work→actor | Hamilton | after (26.5%) | 4 | 0.4 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |

## Per-Prompt Results — variant a2

Rank score: top-1 = 1.0, top-2 = 0.8, top-3 = 0.6, top-4 = 0.4, top-5 = 0.2, not found = 0.0

| Slug | Hop type | Intermediate | Predicted | Rank | Score | Hop found? | Hop features | Mean influence |
|------|----------|--------------|-----------|-----:|------:|------------|--------------|----------------|
| dallas-capital | city→state→capital | Texas | Austin (33.5%) | 1 | 1.0 | ✓ (9 feat, groups: Texas, say Texas, Output: "Texas" (4.1%)) | 9 (0.6%) | 0.3045 |
| sf-capital | city→state→capital | California | Sacramento (29.7%) | 1 | 1.0 | ✓ (11 feat, groups: California, say California, Output: "California" (5.0%)) | 11 (0.7%) | 0.3389 |
| las-vegas-capital | city→state→capital | Nevada | Reno (16.1%) | 3 | 0.6 | ✓ (4 feat, groups: say Nevada, Nevada, Output: "Nevada" (11.8%)) | 4 (0.2%) | 0.3038 |
| oakland-capital | city→state→capital | California | Sacramento (27.7%) | 1 | 1.0 | ✓ (8 feat, groups: California) | 8 (0.5%) | 0.3291 |
| grand-canyon-capital | landmark→state→capital | Arizona | Phoenix (27.3%) | 1 | 1.0 | ✓ (6 feat, groups: Arizona, say Arizona-Tucson) | 6 (0.3%) | 0.3069 |
| miami-capital | city→state→capital | Florida | Tallahassee (42.5%) | 1 | 1.0 | ✓ (8 feat, groups: Florida, Output: "Florida" (3.8%)) | 8 (0.6%) | 0.3251 |
| hollywood-capital | landmark→state→capital | California | Los (19.1%) | 5 | 0.2 | ✓ (7 feat, groups: California, say California, Output: "California" (5.1%)) | 7 (0.4%) | 0.3160 |
| spokane-capital | city→state→capital | Washington | Olympia (21.5%) | 1 | 1.0 | ✓ (4 feat, groups: Washington (state)) | 4 (0.3%) | 0.3142 |
| packers-capital | team→state→capital | Wisconsin | Green (9.0%) | 3 | 0.6 | ✓ (5 feat, groups: Wisconsin place names, say Wisconsin) | 5 (0.2%) | 0.3423 |
| niagara-falls-capital | landmark→state→capital | New York | Buffalo (17.4%) | 2 | 0.8 | ✓ (5 feat, groups: New York (state)) | 5 (0.3%) | 0.3224 |
| east-california-capital | geo→state→capital | Nevada | Sacramento (33.0%) | 3 | 0.6 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3226 |
| saints-capital | team→state→capital | Louisiana | New (12.8%) | 2 | 0.8 | ✓ (6 feat, groups: New Orleans, Louisiana, say Louisiana-Parish) | 6 (0.3%) | 0.3238 |
| memphis-capital | city→state→capital | Tennessee | the (15.6%) | 2 | 0.8 | ✓ (4 feat, groups: say Nashville-Tennessee, Output: "Tennessee" (8.8%)) | 4 (0.3%) | 0.3120 |
| yellowstone-national-park-capital | landmark→state→capital | Wyoming | the (14.5%) | 2 | 0.8 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3391 |
| mount-rushmore-capital | landmark→state→capital | South Dakota | South (13.9%) | 3 | 0.6 | ✓ (1 feat, groups: South Dakota) | 1 (0.1%) | 0.3379 |
| gettysburg-capital | landmark→state→capital | Pennsylvania | Harrisburg (61.2%) | 1 | 1.0 | ✓ (9 feat, groups: Pennsylvania, say Pennsylvania city, Output: "Pennsylvania" (1.2%)) | 9 (0.6%) | 0.3260 |
| orlando-capital | city→state→capital | Florida | Tallahassee (36.6%) | 1 | 1.0 | ✓ (7 feat, groups: Florida) | 7 (0.5%) | 0.3263 |
| albuquerque-capital | city→state→capital | New Mexico | Santa (39.7%) | 1 | 1.0 | ✓ (3 feat, groups: New Mexico, say New Mexico) | 3 (0.2%) | 0.3254 |
| detroit-capital | city→state→capital | Michigan | Lansing (52.3%) | 1 | 1.0 | ✓ (7 feat, groups: Michigan, Output: "Michigan" (7.4%)) | 7 (0.5%) | 0.3328 |
| minneapolis-capital | city→state→capital | Minnesota | St (25.2%) | 2 | 0.8 | ✓ (7 feat, groups: say Minnesota, Minnesota (state), Output: "Minnesota" (9.3%)) | 7 (0.5%) | 0.3267 |
| machu-picchu-language | landmark→country→language | Peru | Que (40.0%) | 2 | 0.8 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3099 |
| forbidden-city-language | landmark→country→language | China | Chinese (32.5%) | 1 | 1.0 | ✓ (9 feat, groups: China-Chinese, say China) | 9 (0.4%) | 0.3279 |
| big-ben-language | landmark→country→language | United Kingdom | English (37.0%) | 1 | 1.0 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3611 |
| colosseum-language | landmark→country→language | Italy | Italian (34.4%) | 1 | 1.0 | ✓ (8 feat, groups: Italy - Italian) | 8 (0.4%) | 0.3461 |
| eiffel-tower-language | landmark→country→language | France | French (56.4%) | 1 | 1.0 | ✓ (8 feat, groups: France) | 8 (0.4%) | 0.3425 |
| taj-mahal-language | landmark→country→language | India | Hindi (23.4%) | 1 | 1.0 | ✓ (3 feat, groups: India) | 3 (0.1%) | 0.3318 |
| great-wall-language | landmark→country→language | China | Chinese (22.3%) | 1 | 1.0 | ✓ (9 feat, groups: China - Chinese, say China-Chinese) | 9 (0.4%) | 0.3249 |
| mount-fuji-language | landmark→country→language | Japan | Japanese (41.2%) | 1 | 1.0 | ✓ (9 feat, groups: Japan - Japanese) | 9 (0.4%) | 0.3001 |
| burj-khalifa-language | landmark→country→language | UAE | Arabic (55.0%) | 1 | 1.0 | ✓ (4 feat, groups: —) | 4 (0.2%) | 0.3556 |
| angkor-wat-langugage | landmark→country→language | Cambodia | Khmer (26.6%) | 1 | 1.0 | ✓ (3 feat, groups: Cambodia - Khmer) | 3 (0.1%) | 0.3136 |
| animal-farm-lang | book→author→country | George Orwell | Great (14.2%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| crucible-country | book→author→country | Arthur Miller | the (17.7%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| alchemist-birth-country | book→author→country | Paulo Coelho | Spain (21.6%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| hamlet-country | book→author→country | William Shakespeare | Denmark (22.9%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| tale-country | book→author→country | Charles Dickens | France (21.4%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| beethoven-fifth-symphony-country | work→composer→country | Ludwig van Beethoven | not (9.3%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| penicillin-country | achievement→scientist→country | Alexander Fleming | the (10.1%) | 5 | 0.2 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| darwin-birth | achievement→scientist→country | Charles Darwin | England (11.1%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| harry-potter-birth | work→author→country | J.K. Rowling | the (11.8%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| relativity-birth | achievement→scientist→country | Albert Einstein | Germany (10.9%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| fdr-president | achievement→president→successor | Franklin D. Roosevelt | the (18.2%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| andrew-johnson | achievement→president→successor | Abraham Lincoln | the (18.5%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| john-adams | achievement→president→successor | George Washington | the (13.7%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| donald-trump | achievement→president→successor | Barack Obama | the (15.7%) | 3 | 0.6 | ✓ (2 feat, groups: —) | 2 (0.1%) | 0.3520 |
| warren-harding | achievement→president→successor | Woodrow Wilson | the (16.0%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| vivaldi-nationality | work→composer→nationality | Antonio Vivaldi | unknown (9.4%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| stone-stars | song→film→actress | La La Land | Emma (8.1%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| gosling-stars | song→film→actor | La La Land | Ryan (16.8%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| da-vinci-painting | work→artist→century | Leonardo da Vinci | the (11.2%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| hamilton-lead-actor | work→actor | Hamilton | after (26.5%) | 4 | 0.4 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |

## Per-Prompt Results — variant a3

Rank score: top-1 = 1.0, top-2 = 0.8, top-3 = 0.6, top-4 = 0.4, top-5 = 0.2, not found = 0.0

| Slug | Hop type | Intermediate | Predicted | Rank | Score | Hop found? | Hop features | Mean influence |
|------|----------|--------------|-----------|-----:|------:|------------|--------------|----------------|
| dallas-capital | city→state→capital | Texas | Austin (33.5%) | 1 | 1.0 | ✓ (9 feat, groups: Texas, say Texas, Output: "Texas" (4.1%)) | 9 (0.6%) | 0.3045 |
| sf-capital | city→state→capital | California | Sacramento (29.7%) | 1 | 1.0 | ✓ (11 feat, groups: California, say California, Output: "California" (5.0%)) | 11 (0.7%) | 0.3389 |
| las-vegas-capital | city→state→capital | Nevada | Reno (16.1%) | 3 | 0.6 | ✓ (4 feat, groups: say Nevada, Nevada, Output: "Nevada" (11.8%)) | 4 (0.2%) | 0.3038 |
| oakland-capital | city→state→capital | California | Sacramento (27.7%) | 1 | 1.0 | ✓ (8 feat, groups: California) | 8 (0.5%) | 0.3291 |
| grand-canyon-capital | landmark→state→capital | Arizona | Phoenix (27.3%) | 1 | 1.0 | ✓ (6 feat, groups: Arizona, say Arizona-Tucson) | 6 (0.3%) | 0.3069 |
| miami-capital | city→state→capital | Florida | Tallahassee (42.5%) | 1 | 1.0 | ✓ (8 feat, groups: Florida, Output: "Florida" (3.8%)) | 8 (0.6%) | 0.3251 |
| hollywood-capital | landmark→state→capital | California | Los (19.1%) | 5 | 0.2 | ✓ (7 feat, groups: California, say California, Output: "California" (5.1%)) | 7 (0.4%) | 0.3160 |
| spokane-capital | city→state→capital | Washington | Olympia (21.5%) | 1 | 1.0 | ✓ (4 feat, groups: Washington (state)) | 4 (0.3%) | 0.3142 |
| packers-capital | team→state→capital | Wisconsin | Green (9.0%) | 3 | 0.6 | ✓ (5 feat, groups: Wisconsin place names, say Wisconsin) | 5 (0.2%) | 0.3423 |
| niagara-falls-capital | landmark→state→capital | New York | Buffalo (17.4%) | 2 | 0.8 | ✓ (5 feat, groups: New York (state)) | 5 (0.3%) | 0.3224 |
| east-california-capital | geo→state→capital | Nevada | Sacramento (33.0%) | 3 | 0.6 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3226 |
| saints-capital | team→state→capital | Louisiana | New (12.8%) | 2 | 0.8 | ✓ (6 feat, groups: New Orleans, Louisiana, say Louisiana-Parish) | 6 (0.3%) | 0.3238 |
| memphis-capital | city→state→capital | Tennessee | the (15.6%) | 2 | 0.8 | ✓ (4 feat, groups: say Nashville-Tennessee, Output: "Tennessee" (8.8%)) | 4 (0.3%) | 0.3120 |
| yellowstone-national-park-capital | landmark→state→capital | Wyoming | the (14.5%) | 2 | 0.8 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3391 |
| mount-rushmore-capital | landmark→state→capital | South Dakota | South (13.9%) | 3 | 0.6 | ✓ (1 feat, groups: South Dakota) | 1 (0.1%) | 0.3379 |
| gettysburg-capital | landmark→state→capital | Pennsylvania | Harrisburg (61.2%) | 1 | 1.0 | ✓ (9 feat, groups: Pennsylvania, say Pennsylvania city, Output: "Pennsylvania" (1.2%)) | 9 (0.6%) | 0.3260 |
| orlando-capital | city→state→capital | Florida | Tallahassee (36.6%) | 1 | 1.0 | ✓ (7 feat, groups: Florida) | 7 (0.5%) | 0.3263 |
| albuquerque-capital | city→state→capital | New Mexico | Santa (39.7%) | 1 | 1.0 | ✓ (3 feat, groups: New Mexico, say New Mexico) | 3 (0.2%) | 0.3254 |
| detroit-capital | city→state→capital | Michigan | Lansing (52.3%) | 1 | 1.0 | ✓ (7 feat, groups: Michigan, Output: "Michigan" (7.4%)) | 7 (0.5%) | 0.3328 |
| minneapolis-capital | city→state→capital | Minnesota | St (25.2%) | 2 | 0.8 | ✓ (7 feat, groups: say Minnesota, Minnesota (state), Output: "Minnesota" (9.3%)) | 7 (0.5%) | 0.3267 |
| machu-picchu-language | landmark→country→language | Peru | Que (40.0%) | 2 | 0.8 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3099 |
| forbidden-city-language | landmark→country→language | China | Chinese (32.5%) | 1 | 1.0 | ✓ (9 feat, groups: China-Chinese, say China) | 9 (0.4%) | 0.3279 |
| big-ben-language | landmark→country→language | United Kingdom | English (37.0%) | 1 | 1.0 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3611 |
| colosseum-language | landmark→country→language | Italy | Italian (34.4%) | 1 | 1.0 | ✓ (8 feat, groups: Italy - Italian) | 8 (0.4%) | 0.3461 |
| eiffel-tower-language | landmark→country→language | France | French (56.4%) | 1 | 1.0 | ✓ (8 feat, groups: France) | 8 (0.4%) | 0.3425 |
| taj-mahal-language | landmark→country→language | India | Hindi (23.4%) | 1 | 1.0 | ✓ (3 feat, groups: India) | 3 (0.1%) | 0.3318 |
| great-wall-language | landmark→country→language | China | Chinese (22.3%) | 1 | 1.0 | ✓ (9 feat, groups: China - Chinese, say China-Chinese) | 9 (0.4%) | 0.3249 |
| mount-fuji-language | landmark→country→language | Japan | Japanese (41.2%) | 1 | 1.0 | ✓ (9 feat, groups: Japan - Japanese) | 9 (0.4%) | 0.3001 |
| burj-khalifa-language | landmark→country→language | UAE | Arabic (55.0%) | 1 | 1.0 | ✓ (4 feat, groups: —) | 4 (0.2%) | 0.3556 |
| angkor-wat-langugage | landmark→country→language | Cambodia | Khmer (26.6%) | 1 | 1.0 | ✓ (3 feat, groups: Cambodia - Khmer) | 3 (0.1%) | 0.3136 |
| animal-farm-lang | book→author→country | George Orwell | Great (14.2%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| crucible-country | book→author→country | Arthur Miller | the (17.7%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| alchemist-birth-country | book→author→country | Paulo Coelho | Spain (21.6%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| hamlet-country | book→author→country | William Shakespeare | Denmark (22.9%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| tale-country | book→author→country | Charles Dickens | France (21.4%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| beethoven-fifth-symphony-country | work→composer→country | Ludwig van Beethoven | not (9.3%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| penicillin-country | achievement→scientist→country | Alexander Fleming | the (10.1%) | 5 | 0.2 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| darwin-birth | achievement→scientist→country | Charles Darwin | England (11.1%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| harry-potter-birth | work→author→country | J.K. Rowling | the (11.8%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| relativity-birth | achievement→scientist→country | Albert Einstein | Germany (10.9%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| fdr-president | achievement→president→successor | Franklin D. Roosevelt | the (18.2%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| andrew-johnson | achievement→president→successor | Abraham Lincoln | the (18.5%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| john-adams | achievement→president→successor | George Washington | the (13.7%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| donald-trump | achievement→president→successor | Barack Obama | the (15.7%) | 3 | 0.6 | ✓ (2 feat, groups: —) | 2 (0.1%) | 0.3520 |
| warren-harding | achievement→president→successor | Woodrow Wilson | the (16.0%) | 3 | 0.6 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| vivaldi-nationality | work→composer→nationality | Antonio Vivaldi | unknown (9.4%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| stone-stars | song→film→actress | La La Land | Emma (8.1%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| gosling-stars | song→film→actor | La La Land | Ryan (16.8%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| da-vinci-painting | work→artist→century | Leonardo da Vinci | the (11.2%) | — | 0.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| hamilton-lead-actor | work→actor | Hamilton | after (26.5%) | 4 | 0.4 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |

## Spotlight: Model Wrong but Intermediate Hop Present

These are the most interpretability-interesting cases — the model encoded the intermediate concept but still predicted incorrectly.

### las-vegas-capital (a0)
- Prompt type: city→state→capital
- Intermediate concept: **Nevada**
- Correct answer: Carson City
- Model predicted: **Reno** (16.1%)
- Hop features: 4 (0.2% of transcoder nodes)
- Hop in supernode groups: ['say Nevada', 'Nevada', 'Output: "Nevada" (11.8%)']
- Mean influence of hop features: 0.3038

### las-vegas-capital (a1)
- Prompt type: city→state→capital
- Intermediate concept: **Nevada**
- Correct answer: Carson City
- Model predicted: **Reno** (16.1%)
- Hop features: 4 (0.2% of transcoder nodes)
- Hop in supernode groups: ['say Nevada', 'Nevada', 'Output: "Nevada" (11.8%)']
- Mean influence of hop features: 0.3038

### las-vegas-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Nevada**
- Correct answer: Carson City
- Model predicted: **Reno** (16.1%)
- Hop features: 4 (0.2% of transcoder nodes)
- Hop in supernode groups: ['say Nevada', 'Nevada', 'Output: "Nevada" (11.8%)']
- Mean influence of hop features: 0.3038

### las-vegas-capital (a3)
- Prompt type: city→state→capital
- Intermediate concept: **Nevada**
- Correct answer: Carson City
- Model predicted: **Reno** (16.1%)
- Hop features: 4 (0.2% of transcoder nodes)
- Hop in supernode groups: ['say Nevada', 'Nevada', 'Output: "Nevada" (11.8%)']
- Mean influence of hop features: 0.3038

### hollywood-capital (a0)
- Prompt type: landmark→state→capital
- Intermediate concept: **California**
- Correct answer: Sacramento
- Model predicted: **Los** (19.1%)
- Hop features: 7 (0.4% of transcoder nodes)
- Hop in supernode groups: ['California', 'say California', 'Output: "California" (5.1%)']
- Mean influence of hop features: 0.3160

### hollywood-capital (a1)
- Prompt type: landmark→state→capital
- Intermediate concept: **California**
- Correct answer: Sacramento
- Model predicted: **Los** (19.1%)
- Hop features: 7 (0.4% of transcoder nodes)
- Hop in supernode groups: ['California', 'say California', 'Output: "California" (5.1%)']
- Mean influence of hop features: 0.3160

### hollywood-capital (a2)
- Prompt type: landmark→state→capital
- Intermediate concept: **California**
- Correct answer: Sacramento
- Model predicted: **Los** (19.1%)
- Hop features: 7 (0.4% of transcoder nodes)
- Hop in supernode groups: ['California', 'say California', 'Output: "California" (5.1%)']
- Mean influence of hop features: 0.3160

### hollywood-capital (a3)
- Prompt type: landmark→state→capital
- Intermediate concept: **California**
- Correct answer: Sacramento
- Model predicted: **Los** (19.1%)
- Hop features: 7 (0.4% of transcoder nodes)
- Hop in supernode groups: ['California', 'say California', 'Output: "California" (5.1%)']
- Mean influence of hop features: 0.3160

### packers-capital (a0)
- Prompt type: team→state→capital
- Intermediate concept: **Wisconsin**
- Correct answer: Madison
- Model predicted: **Green** (9.0%)
- Hop features: 5 (0.2% of transcoder nodes)
- Hop in supernode groups: ['Wisconsin place names', 'say Wisconsin']
- Mean influence of hop features: 0.3423

### packers-capital (a1)
- Prompt type: team→state→capital
- Intermediate concept: **Wisconsin**
- Correct answer: Madison
- Model predicted: **Green** (9.0%)
- Hop features: 5 (0.2% of transcoder nodes)
- Hop in supernode groups: ['Wisconsin place names', 'say Wisconsin']
- Mean influence of hop features: 0.3423

### packers-capital (a2)
- Prompt type: team→state→capital
- Intermediate concept: **Wisconsin**
- Correct answer: Madison
- Model predicted: **Green** (9.0%)
- Hop features: 5 (0.2% of transcoder nodes)
- Hop in supernode groups: ['Wisconsin place names', 'say Wisconsin']
- Mean influence of hop features: 0.3423

### packers-capital (a3)
- Prompt type: team→state→capital
- Intermediate concept: **Wisconsin**
- Correct answer: Madison
- Model predicted: **Green** (9.0%)
- Hop features: 5 (0.2% of transcoder nodes)
- Hop in supernode groups: ['Wisconsin place names', 'say Wisconsin']
- Mean influence of hop features: 0.3423

### niagara-falls-capital (a0)
- Prompt type: landmark→state→capital
- Intermediate concept: **New York**
- Correct answer: Albany
- Model predicted: **Buffalo** (17.4%)
- Hop features: 5 (0.3% of transcoder nodes)
- Hop in supernode groups: ['New York (state)']
- Mean influence of hop features: 0.3224

### niagara-falls-capital (a1)
- Prompt type: landmark→state→capital
- Intermediate concept: **New York**
- Correct answer: Albany
- Model predicted: **Buffalo** (17.4%)
- Hop features: 5 (0.3% of transcoder nodes)
- Hop in supernode groups: ['New York (state)']
- Mean influence of hop features: 0.3224

### niagara-falls-capital (a2)
- Prompt type: landmark→state→capital
- Intermediate concept: **New York**
- Correct answer: Albany
- Model predicted: **Buffalo** (17.4%)
- Hop features: 5 (0.3% of transcoder nodes)
- Hop in supernode groups: ['New York (state)']
- Mean influence of hop features: 0.3224

### niagara-falls-capital (a3)
- Prompt type: landmark→state→capital
- Intermediate concept: **New York**
- Correct answer: Albany
- Model predicted: **Buffalo** (17.4%)
- Hop features: 5 (0.3% of transcoder nodes)
- Hop in supernode groups: ['New York (state)']
- Mean influence of hop features: 0.3224

### east-california-capital (a0)
- Prompt type: geo→state→capital
- Intermediate concept: **Nevada**
- Correct answer: Carson City
- Model predicted: **Sacramento** (33.0%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3226

### east-california-capital (a1)
- Prompt type: geo→state→capital
- Intermediate concept: **Nevada**
- Correct answer: Carson City
- Model predicted: **Sacramento** (33.0%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3226

### east-california-capital (a2)
- Prompt type: geo→state→capital
- Intermediate concept: **Nevada**
- Correct answer: Carson City
- Model predicted: **Sacramento** (33.0%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3226

### east-california-capital (a3)
- Prompt type: geo→state→capital
- Intermediate concept: **Nevada**
- Correct answer: Carson City
- Model predicted: **Sacramento** (33.0%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3226

### saints-capital (a0)
- Prompt type: team→state→capital
- Intermediate concept: **Louisiana**
- Correct answer: Baton Rouge
- Model predicted: **New** (12.8%)
- Hop features: 6 (0.3% of transcoder nodes)
- Hop in supernode groups: ['New Orleans, Louisiana', 'say Louisiana-Parish']
- Mean influence of hop features: 0.3238

### saints-capital (a1)
- Prompt type: team→state→capital
- Intermediate concept: **Louisiana**
- Correct answer: Baton Rouge
- Model predicted: **New** (12.8%)
- Hop features: 6 (0.3% of transcoder nodes)
- Hop in supernode groups: ['New Orleans, Louisiana', 'say Louisiana-Parish']
- Mean influence of hop features: 0.3238

### saints-capital (a2)
- Prompt type: team→state→capital
- Intermediate concept: **Louisiana**
- Correct answer: Baton Rouge
- Model predicted: **New** (12.8%)
- Hop features: 6 (0.3% of transcoder nodes)
- Hop in supernode groups: ['New Orleans, Louisiana', 'say Louisiana-Parish']
- Mean influence of hop features: 0.3238

### saints-capital (a3)
- Prompt type: team→state→capital
- Intermediate concept: **Louisiana**
- Correct answer: Baton Rouge
- Model predicted: **New** (12.8%)
- Hop features: 6 (0.3% of transcoder nodes)
- Hop in supernode groups: ['New Orleans, Louisiana', 'say Louisiana-Parish']
- Mean influence of hop features: 0.3238

### memphis-capital (a0)
- Prompt type: city→state→capital
- Intermediate concept: **Tennessee**
- Correct answer: Nashville
- Model predicted: **the** (15.6%)
- Hop features: 4 (0.3% of transcoder nodes)
- Hop in supernode groups: ['say Nashville-Tennessee', 'Output: "Tennessee" (8.8%)']
- Mean influence of hop features: 0.3120

### memphis-capital (a1)
- Prompt type: city→state→capital
- Intermediate concept: **Tennessee**
- Correct answer: Nashville
- Model predicted: **the** (15.6%)
- Hop features: 4 (0.3% of transcoder nodes)
- Hop in supernode groups: ['say Nashville-Tennessee', 'Output: "Tennessee" (8.8%)']
- Mean influence of hop features: 0.3120

### memphis-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Tennessee**
- Correct answer: Nashville
- Model predicted: **the** (15.6%)
- Hop features: 4 (0.3% of transcoder nodes)
- Hop in supernode groups: ['say Nashville-Tennessee', 'Output: "Tennessee" (8.8%)']
- Mean influence of hop features: 0.3120

### memphis-capital (a3)
- Prompt type: city→state→capital
- Intermediate concept: **Tennessee**
- Correct answer: Nashville
- Model predicted: **the** (15.6%)
- Hop features: 4 (0.3% of transcoder nodes)
- Hop in supernode groups: ['say Nashville-Tennessee', 'Output: "Tennessee" (8.8%)']
- Mean influence of hop features: 0.3120

### yellowstone-national-park-capital (a0)
- Prompt type: landmark→state→capital
- Intermediate concept: **Wyoming**
- Correct answer: Cheyenne
- Model predicted: **the** (14.5%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3391

### yellowstone-national-park-capital (a1)
- Prompt type: landmark→state→capital
- Intermediate concept: **Wyoming**
- Correct answer: Cheyenne
- Model predicted: **the** (14.5%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3391

### yellowstone-national-park-capital (a2)
- Prompt type: landmark→state→capital
- Intermediate concept: **Wyoming**
- Correct answer: Cheyenne
- Model predicted: **the** (14.5%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3391

### yellowstone-national-park-capital (a3)
- Prompt type: landmark→state→capital
- Intermediate concept: **Wyoming**
- Correct answer: Cheyenne
- Model predicted: **the** (14.5%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3391

### mount-rushmore-capital (a0)
- Prompt type: landmark→state→capital
- Intermediate concept: **South Dakota**
- Correct answer: Pierre
- Model predicted: **South** (13.9%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: ['South Dakota']
- Mean influence of hop features: 0.3379

### mount-rushmore-capital (a1)
- Prompt type: landmark→state→capital
- Intermediate concept: **South Dakota**
- Correct answer: Pierre
- Model predicted: **South** (13.9%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: ['South Dakota']
- Mean influence of hop features: 0.3379

### mount-rushmore-capital (a2)
- Prompt type: landmark→state→capital
- Intermediate concept: **South Dakota**
- Correct answer: Pierre
- Model predicted: **South** (13.9%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: ['South Dakota']
- Mean influence of hop features: 0.3379

### mount-rushmore-capital (a3)
- Prompt type: landmark→state→capital
- Intermediate concept: **South Dakota**
- Correct answer: Pierre
- Model predicted: **South** (13.9%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: ['South Dakota']
- Mean influence of hop features: 0.3379

### minneapolis-capital (a0)
- Prompt type: city→state→capital
- Intermediate concept: **Minnesota**
- Correct answer: Saint Paul
- Model predicted: **St** (25.2%)
- Hop features: 7 (0.5% of transcoder nodes)
- Hop in supernode groups: ['say Minnesota', 'Minnesota (state)', 'Output: "Minnesota" (9.3%)']
- Mean influence of hop features: 0.3267

### minneapolis-capital (a1)
- Prompt type: city→state→capital
- Intermediate concept: **Minnesota**
- Correct answer: Saint Paul
- Model predicted: **St** (25.2%)
- Hop features: 7 (0.5% of transcoder nodes)
- Hop in supernode groups: ['say Minnesota', 'Minnesota (state)', 'Output: "Minnesota" (9.3%)']
- Mean influence of hop features: 0.3267

### minneapolis-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Minnesota**
- Correct answer: Saint Paul
- Model predicted: **St** (25.2%)
- Hop features: 7 (0.5% of transcoder nodes)
- Hop in supernode groups: ['say Minnesota', 'Minnesota (state)', 'Output: "Minnesota" (9.3%)']
- Mean influence of hop features: 0.3267

### minneapolis-capital (a3)
- Prompt type: city→state→capital
- Intermediate concept: **Minnesota**
- Correct answer: Saint Paul
- Model predicted: **St** (25.2%)
- Hop features: 7 (0.5% of transcoder nodes)
- Hop in supernode groups: ['say Minnesota', 'Minnesota (state)', 'Output: "Minnesota" (9.3%)']
- Mean influence of hop features: 0.3267

### machu-picchu-language (a0)
- Prompt type: landmark→country→language
- Intermediate concept: **Peru**
- Correct answer: Spanish
- Model predicted: **Que** (40.0%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3099

### machu-picchu-language (a1)
- Prompt type: landmark→country→language
- Intermediate concept: **Peru**
- Correct answer: Spanish
- Model predicted: **Que** (40.0%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3099

### machu-picchu-language (a2)
- Prompt type: landmark→country→language
- Intermediate concept: **Peru**
- Correct answer: Spanish
- Model predicted: **Que** (40.0%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3099

### machu-picchu-language (a3)
- Prompt type: landmark→country→language
- Intermediate concept: **Peru**
- Correct answer: Spanish
- Model predicted: **Que** (40.0%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3099

### donald-trump (a0)
- Prompt type: achievement→president→successor
- Intermediate concept: **Barack Obama**
- Correct answer: Donald Trump
- Model predicted: **the** (15.7%)
- Hop features: 2 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3520

### donald-trump (a1)
- Prompt type: achievement→president→successor
- Intermediate concept: **Barack Obama**
- Correct answer: Donald Trump
- Model predicted: **the** (15.7%)
- Hop features: 2 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3520

### donald-trump (a2)
- Prompt type: achievement→president→successor
- Intermediate concept: **Barack Obama**
- Correct answer: Donald Trump
- Model predicted: **the** (15.7%)
- Hop features: 2 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3520

### donald-trump (a3)
- Prompt type: achievement→president→successor
- Intermediate concept: **Barack Obama**
- Correct answer: Donald Trump
- Model predicted: **the** (15.7%)
- Hop features: 2 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3520

