# Intermediate Hop Analysis

Variants analysed: a2  
Prompts analysed: 50  
Total graph runs: 50

## Aggregate Stats

| Metric | Count | Fraction |
|--------|-------|----------|
| Model correct (top-1) | 35 | 70.0% |
| Correct answer in top-5 | 50 | 100.0% |
| Mean rank score (0–1) | — | 0.91 |
| Intermediate hop found (any) | 48 | 96.0% |
| Hop found in feature clerps | 48 | 96.0% |
| Hop found in supernode groups | 43 | 86.0% |

## Correctness × Hop Presence

| | Hop found | No hop found |
|---|---|---|
| **Model correct** | 34 | 1 |
| **Model wrong** | 14 | 1 |

> When model is **wrong**: hop found in 93.3% of cases  
> When model is **correct**: hop found in 97.1% of cases

## Per-Prompt Results — variant a2

Rank score: top-1 = 1.0, top-2 = 0.8, top-3 = 0.6, top-4 = 0.4, top-5 = 0.2, not found = 0.0

| Slug | Hop type | Intermediate | Predicted | Rank | Score | Hop found? | Hop features | Mean influence |
|------|----------|--------------|-----------|-----:|------:|------------|--------------|----------------|
| birmingham-capital | city→state→capital | Alabama | the (17.4%) | 3 | 0.6 | ✓ (3 feat, groups: Alabama) | 3 (0.2%) | 0.3269 |
| anchorage-capital | city→state→capital | Alaska | Juneau (32.5%) | 1 | 1.0 | ✓ (5 feat, groups: Alaska, say Alaska, Output: "Alaska" (4.1%)) | 5 (0.3%) | 0.3085 |
| grand-canyon-capital | landmark→state→capital | Arizona | Phoenix (27.3%) | 1 | 1.0 | ✓ (4 feat, groups: Arizona) | 4 (0.2%) | 0.3250 |
| fort-smith-capital | city→state→capital | Arkansas | Fort (12.9%) | 3 | 0.6 | ✓ (4 feat, groups: Arkansas and Little Rock, say Arkansas) | 4 (0.2%) | 0.3158 |
| los-angeles-capital | city→state→capital | California | Sacramento (29.3%) | 1 | 1.0 | ✓ (17 feat, groups: Southern California places, California, say California, Output: "California" (6.6%)) | 17 (1.1%) | 0.3359 |
| colorado-springs-capital | city→state→capital | Colorado | Denver (48.1%) | 1 | 1.0 | ✓ (7 feat, groups: Colorado places, say Colorado place, Emb: "Colorado", Output: "Colorado" (13.1%)) | 7 (0.4%) | 0.3205 |
| new-haven-capital | city→state→capital | Connecticut | Hartford (33.2%) | 1 | 1.0 | ✓ (6 feat, groups: Connecticut, Output: "Connecticut" (6.6%)) | 6 (0.4%) | 0.3195 |
| wilmington-capital | city→state→capital | Delaware | Raleigh (21.2%) | 3 | 0.6 | ✓ (2 feat, groups: Delaware) | 2 (0.1%) | 0.3588 |
| miami-capital | city→state→capital | Florida | Tallahassee (42.5%) | 1 | 1.0 | ✓ (6 feat, groups: Florida, say Florida, Output: "Florida" (3.8%)) | 6 (0.4%) | 0.3051 |
| savannah-capital | city→state→capital | Georgia | Atlanta (14.2%) | 1 | 1.0 | ✓ (6 feat, groups: Georgia (state), Georgia (general)) | 6 (0.4%) | 0.3135 |
| pearl-harbor-capital | landmark→state→capital | Hawaii | Honolulu (50.8%) | 1 | 1.0 | ✓ (5 feat, groups: Hawaii, Output: "Hawaii" (7.2%)) | 5 (0.3%) | 0.3138 |
| twin-falls-capital | city→state→capital | Idaho | Boise (18.2%) | 1 | 1.0 | ✓ (1 feat, groups: Idaho region places, Output: "Idaho" (8.2%)) | 1 (0.1%) | 0.3608 |
| chicago-capital | city→state→capital | Illinois | Illinois (25.3%) | 2 | 0.8 | ✓ (10 feat, groups: Illinois, say Illinois, Output: "Illinois" (25.3%) [target]) | 10 (0.7%) | 0.3208 |
| south-bend-capital | city→state→capital | Indiana | Indianapolis (19.6%) | 1 | 1.0 | ✓ (1 feat, groups: Indiana, Output: "Indiana" (5.9%)) | 1 (0.1%) | 0.3098 |
| cedar-rapids-capital | city→state→capital | Iowa | Iowa (34.9%) | 2 | 0.8 | ✓ (3 feat, groups: Iowa, Output: "Iowa" (34.9%) [target]) | 3 (0.2%) | 0.3004 |
| wichita-capital | city→state→capital | Kansas | Topeka (18.1%) | 1 | 1.0 | ✓ (4 feat, groups: say Kansas, Kansas, Output: "Kansas" (12.2%)) | 4 (0.3%) | 0.3157 |
| louisville-capital | city→state→capital | Kentucky | Frankfort (33.6%) | 1 | 1.0 | ✓ (2 feat, groups: Kentucky, Output: "Kentucky" (9.3%)) | 2 (0.1%) | 0.3110 |
| new-orleans-capital | city→state→capital | Louisiana | Baton (41.9%) | 1 | 1.0 | ✓ (5 feat, groups: Louisiana, Output: "Louisiana" (9.7%)) | 5 (0.3%) | 0.3021 |
| bar-harbor-capital | city→state→capital | Maine | Augusta (34.8%) | 1 | 1.0 | ✓ (5 feat, groups: Maine place names, Maine (U.S. state)) | 5 (0.3%) | 0.3165 |
| baltimore-capital | city→state→capital | Maryland | Annapolis (43.3%) | 1 | 1.0 | ✓ (2 feat, groups: Maryland, Output: "Maryland" (5.7%)) | 2 (0.1%) | 0.2765 |
| cambridge-capital | city→state→capital | Massachusetts | Boston (27.5%) | 1 | 1.0 | ✓ (5 feat, groups: Massachusetts, say Massachusetts, Output: "Massachusetts" (4.6%)) | 5 (0.3%) | 0.3482 |
| detroit-capital | city→state→capital | Michigan | Lansing (52.3%) | 1 | 1.0 | ✓ (7 feat, groups: Michigan, Output: "Michigan" (7.4%)) | 7 (0.5%) | 0.3243 |
| minneapolis-capital | city→state→capital | Minnesota | St (25.2%) | 2 | 0.8 | ✓ (4 feat, groups: Minnesota, say Minnesota, Output: "Minnesota" (9.3%)) | 4 (0.3%) | 0.3168 |
| biloxi-capital | city→state→capital | Mississippi | Jackson (18.2%) | 1 | 1.0 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.2816 |
| st-louis-capital | city→state→capital | Missouri | Jefferson (27.1%) | 1 | 1.0 | ✓ (7 feat, groups: Missouri) | 7 (0.4%) | 0.3446 |
| billings-capital | city→state→capital | Montana | Helena (33.6%) | 1 | 1.0 | ✓ (4 feat, groups: say Montana) | 4 (0.3%) | 0.3179 |
| omaha-capital | city→state→capital | Nebraska | Lincoln (45.3%) | 1 | 1.0 | ✓ (2 feat, groups: Nebraska, Output: "Nebraska" (7.7%)) | 2 (0.1%) | 0.2887 |
| las-vegas-capital | city→state→capital | Nevada | Reno (16.1%) | 3 | 0.6 | ✓ (3 feat, groups: Las Vegas - Nevada, say Nevada, Output: "Nevada" (11.8%)) | 3 (0.2%) | 0.3144 |
| nashua-capital | city→state→capital | New Hampshire | Concord (43.9%) | 1 | 1.0 | ✓ (1 feat, groups: —) | 1 (0.1%) | 0.3016 |
| atlantic-city-capital | city→state→capital | New Jersey | Atlantic (17.6%) | 2 | 0.8 | ✓ (7 feat, groups: New Jersey) | 7 (0.4%) | 0.3343 |
| albuquerque-capital | city→state→capital | New Mexico | Santa (39.7%) | 1 | 1.0 | ✓ (1 feat, groups: say New Mexico) | 1 (0.1%) | 0.2758 |
| niagara-falls-capital | city→state→capital | New York | Buffalo (17.4%) | 2 | 0.8 | ✓ (4 feat, groups: New York State, Upstate New York) | 4 (0.2%) | 0.3342 |
| charlotte-capital | city→state→capital | North Carolina | Raleigh (38.3%) | 1 | 1.0 | ✓ (5 feat, groups: North Carolina) | 5 (0.3%) | 0.3235 |
| fargo-capital | city→state→capital | North Dakota | Bismarck (23.2%) | 1 | 1.0 | ✓ (2 feat, groups: —) | 2 (0.1%) | 0.3233 |
| cleveland-capital | city→state→capital | Ohio | Columbus (31.9%) | 1 | 1.0 | ✓ (2 feat, groups: Ohio (state), Output: "Ohio" (6.3%)) | 2 (0.1%) | 0.3088 |
| tulsa-capital | city→state→capital | Oklahoma | Oklahoma (41.4%) | 1 | 1.0 | ✓ (3 feat, groups: Oklahoma, Output: "Oklahoma" (41.4%) [target]) | 3 (0.2%) | 0.3059 |
| portland-or-capital | city→state→capital | Oregon | Salem (22.2%) | 1 | 1.0 | ✓ (3 feat, groups: say Oregon, Oregon) | 3 (0.2%) | 0.3151 |
| philadelphia-capital | city→state→capital | Pennsylvania | Harrisburg (45.3%) | 1 | 1.0 | ✓ (4 feat, groups: Pennsylvania, Output: "Pennsylvania" (5.4%)) | 4 (0.3%) | 0.3001 |
| newport-capital | city→state→capital | Rhode Island | the (15.1%) | 3 | 0.6 | ✓ (2 feat, groups: —) | 2 (0.1%) | 0.3426 |
| myrtle-beach-capital | city→state→capital | South Carolina | Columbia (24.2%) | 1 | 1.0 | ✓ (2 feat, groups: South Carolina) | 2 (0.1%) | 0.3190 |
| mount-rushmore-capital | landmark→state→capital | South Dakota | South (13.9%) | 3 | 0.6 | ✓ (1 feat, groups: say South Dakota) | 1 (0.1%) | 0.3379 |
| memphis-capital | city→state→capital | Tennessee | the (15.6%) | 2 | 0.8 | ✓ (4 feat, groups: Tennessee, say Tennessee-Nashville, Output: "Tennessee" (8.8%)) | 4 (0.3%) | 0.3137 |
| dallas-capital | city→state→capital | Texas | Austin (33.5%) | 1 | 1.0 | ✓ (9 feat, groups: Texas, say Texas, Output: "Texas" (4.1%)) | 9 (0.6%) | 0.3045 |
| provo-capital | city→state→capital | Utah | Provo (23.6%) | 2 | 0.8 | ✓ (4 feat, groups: Utah and Mormonism, Output: "Utah" (7.5%)) | 4 (0.3%) | 0.2950 |
| burlington-vt-capital | city→state→capital | Vermont | the (12.5%) | 2 | 0.8 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |
| virginia-beach-capital | city→state→capital | Virginia | Richmond (24.4%) | 1 | 1.0 | ✓ (1 feat, groups: say Virginia, Emb: "Virginia") | 1 (0.1%) | 0.2513 |
| seattle-capital | city→state→capital | Washington | Olympia (60.2%) | 1 | 1.0 | ✓ (7 feat, groups: Washington State, say Washington DC, Output: "Washington" (4.5%)) | 7 (0.5%) | 0.3368 |
| morgantown-capital | city→state→capital | West Virginia | Morgan (18.7%) | 3 | 0.6 | ✓ (2 feat, groups: —) | 2 (0.1%) | 0.3453 |
| milwaukee-capital | city→state→capital | Wisconsin | Madison (24.1%) | 1 | 1.0 | ✓ (6 feat, groups: Wisconsin, say Wisconsin, Output: "Wisconsin" (10.6%)) | 6 (0.4%) | 0.3042 |
| casper-capital | city→state→capital | Wyoming | Cheyenne (33.3%) | 1 | 1.0 | ✗ (0 feat, groups: —) | 0 (0.0%) | 0.0000 |

## Spotlight: Model Wrong but Intermediate Hop Present

These are the most interpretability-interesting cases — the model encoded the intermediate concept but still predicted incorrectly.

### birmingham-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Alabama**
- Correct answer: Montgomery
- Model predicted: **the** (17.4%)
- Hop features: 3 (0.2% of transcoder nodes)
- Hop in supernode groups: ['Alabama']
- Mean influence of hop features: 0.3269

### fort-smith-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Arkansas**
- Correct answer: Little Rock
- Model predicted: **Fort** (12.9%)
- Hop features: 4 (0.2% of transcoder nodes)
- Hop in supernode groups: ['Arkansas and Little Rock', 'say Arkansas']
- Mean influence of hop features: 0.3158

### wilmington-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Delaware**
- Correct answer: Dover
- Model predicted: **Raleigh** (21.2%)
- Hop features: 2 (0.1% of transcoder nodes)
- Hop in supernode groups: ['Delaware']
- Mean influence of hop features: 0.3588

### chicago-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Illinois**
- Correct answer: Springfield
- Model predicted: **Illinois** (25.3%)
- Hop features: 10 (0.7% of transcoder nodes)
- Hop in supernode groups: ['Illinois', 'say Illinois', 'Output: "Illinois" (25.3%) [target]']
- Mean influence of hop features: 0.3208

### cedar-rapids-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Iowa**
- Correct answer: Des Moines
- Model predicted: **Iowa** (34.9%)
- Hop features: 3 (0.2% of transcoder nodes)
- Hop in supernode groups: ['Iowa', 'Output: "Iowa" (34.9%) [target]']
- Mean influence of hop features: 0.3004

### minneapolis-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Minnesota**
- Correct answer: Saint Paul
- Model predicted: **St** (25.2%)
- Hop features: 4 (0.3% of transcoder nodes)
- Hop in supernode groups: ['Minnesota', 'say Minnesota', 'Output: "Minnesota" (9.3%)']
- Mean influence of hop features: 0.3168

### las-vegas-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Nevada**
- Correct answer: Carson City
- Model predicted: **Reno** (16.1%)
- Hop features: 3 (0.2% of transcoder nodes)
- Hop in supernode groups: ['Las Vegas - Nevada', 'say Nevada', 'Output: "Nevada" (11.8%)']
- Mean influence of hop features: 0.3144

### atlantic-city-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **New Jersey**
- Correct answer: Trenton
- Model predicted: **Atlantic** (17.6%)
- Hop features: 7 (0.4% of transcoder nodes)
- Hop in supernode groups: ['New Jersey']
- Mean influence of hop features: 0.3343

### niagara-falls-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **New York**
- Correct answer: Albany
- Model predicted: **Buffalo** (17.4%)
- Hop features: 4 (0.2% of transcoder nodes)
- Hop in supernode groups: ['New York State', 'Upstate New York']
- Mean influence of hop features: 0.3342

### newport-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Rhode Island**
- Correct answer: Providence
- Model predicted: **the** (15.1%)
- Hop features: 2 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3426

### mount-rushmore-capital (a2)
- Prompt type: landmark→state→capital
- Intermediate concept: **South Dakota**
- Correct answer: Pierre
- Model predicted: **South** (13.9%)
- Hop features: 1 (0.1% of transcoder nodes)
- Hop in supernode groups: ['say South Dakota']
- Mean influence of hop features: 0.3379

### memphis-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Tennessee**
- Correct answer: Nashville
- Model predicted: **the** (15.6%)
- Hop features: 4 (0.3% of transcoder nodes)
- Hop in supernode groups: ['Tennessee', 'say Tennessee-Nashville', 'Output: "Tennessee" (8.8%)']
- Mean influence of hop features: 0.3137

### provo-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **Utah**
- Correct answer: Salt Lake City
- Model predicted: **Provo** (23.6%)
- Hop features: 4 (0.3% of transcoder nodes)
- Hop in supernode groups: ['Utah and Mormonism', 'Output: "Utah" (7.5%)']
- Mean influence of hop features: 0.2950

### morgantown-capital (a2)
- Prompt type: city→state→capital
- Intermediate concept: **West Virginia**
- Correct answer: Charleston
- Model predicted: **Morgan** (18.7%)
- Hop features: 2 (0.1% of transcoder nodes)
- Hop in supernode groups: []
- Mean influence of hop features: 0.3453

