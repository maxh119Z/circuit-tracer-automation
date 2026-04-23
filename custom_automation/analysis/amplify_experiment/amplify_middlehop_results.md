# Amplify Middle-Hop Experiment Results

Variants: a2  
Candidates: 23  
Amplify factor: 2.0×  

## Summary

| Metric | Value |
|--------|-------|
| Candidates with middlehop features found | 18 / 23 |
| Flipped to correct after amplification | 2 / 18 (11.1%) |
| Flipped to middlehop (over-amplified) | 1 / 18 (5.6%) |
| Correct-answer rank improved | 9 / 18 (50.0%) |
| Mean correct-answer prob gain | +0.0103 |

## Per-Prompt Results

| Slug | Intermediate | Correct | Predicted | # Amp | Flipped? | Rank Δ | Prob gain | Baseline top-5 | Amplified top-5 |
|------|-------------|---------|-----------|-------|----------|--------|-----------|----------------|----------------|
| birmingham-capital | Alabama | Montgomery | the | 4 | ✗ | -1 | +0.0023 | the(18.5%), Birmingham(9.9%), (8.7%), called(5.3%), located(5.3%) | the(18.8%), located(10.1%), (7.9%), called(4.8%), Birmingham(4.8%) |
| fort-smith-capital | Arkansas | Little Rock | Fort | 2 | ✗ | -3 | +0.0625 | Fort(13.0%), the(10.2%), (8.9%), Little(7.9%), Fayetteville(7.0%) | Little(14.2%), Fayetteville(11.0%), Fort(11.0%), the(9.8%), (7.6%) |
| wilmington-capital | Delaware | Dover | Raleigh | 2 | ✓ | -2 | +0.1309 | Raleigh(20.2%), the(9.5%), Dover(8.4%), Wilmington(6.5%), (5.8%) | Dover(21.5%), Raleigh(13.1%), Wilmington(9.0%), the(9.0%), Delaware(6.2%) |
| chicago-capital | Illinois | Springfield | Illinois | 5 | ✓ | -1 | +0.1123 | Illinois(25.8%), Springfield(13.8%), the(12.2%), (7.4%), called(4.0%) | Springfield(25.0%), the(17.2%), (7.1%), called(5.6%), Illinois(4.9%) |
| cedar-rapids-capital | Iowa | Des Moines | Iowa | 3 | ✗ | -1 | +0.0918 | Iowa(33.2%), Des(22.9%), the(5.8%), (5.1%), Cedar(4.0%) | Des(32.0%), Iowa(19.4%), the(8.1%), (4.9%), Cedar(3.0%) |
| minneapolis-capital | Minnesota | Saint Paul | St | 3 | ✗ | +1 | +0.0107 | St(24.3%), the(14.7%), Saint(14.7%), Minnesota(10.2%), Minneapolis(5.4%) | St(23.0%), the(15.8%), Saint(15.8%), Minnesota(9.6%), Minneapolis(5.2%) |
| las-vegas-capital | Nevada | Carson City | Reno | 3 | ✗ | +0 | +0.0039 | Reno(17.5%), Nevada(12.0%), (9.4%), Carson(9.4%), the(9.4%) | Nevada(18.3%), Reno(16.1%), Carson(9.8%), (8.6%), the(8.6%) |
| atlantic-city-capital | New Jersey | Trenton | Atlantic | 8 | ✗ | +4 | -0.1040 | Atlantic(17.6%), Trenton(15.5%), (9.4%), the(7.3%), New(3.9%) | Atlantic(15.7%), (13.9%), the(8.4%), New(5.8%), considered(5.8%) |
| newport-capital | Rhode Island | Providence | the | 2 | ✗ | -2 | +0.0491 | the(15.0%), Newport(9.1%), (6.2%), Providence(3.0%), Harrisburg(3.0%) | the(14.7%), Newport(13.0%), Providence(7.9%), (5.4%), Rhode(3.7%) |
| mount-rushmore-capital | South Dakota | Pierre | South | 1 | ✗ | -3 | +0.0288 | South(12.8%), :(8.8%), (7.8%), the(7.8%), (6.8%) | South(16.0%), Pierre(9.7%), the(7.6%), :(6.7%), (6.7%) |
| memphis-capital | Tennessee | Nashville | the | 2 | ✗ | +3 | -0.1001 | the(16.2%), Nashville(14.3%), (11.1%), Memphis(9.8%), Tennessee(8.7%) | the(24.4%), (19.0%), Jackson(5.4%), :(4.2%), Nashville(4.2%) |
| burlington-vt-capital | Vermont | Montpelier | the | 4 | ✗ | -2 | +0.0422 | the(12.3%), (6.6%), Mont(5.8%), Madison(5.1%), Burlington(4.5%) | the(10.1%), Mont(10.1%), Burlington(7.8%), Madison(6.9%), Des(4.8%) |
| morgantown-capital | West Virginia | Charleston | Morgan | 3 | ✗ | +0 | -0.0054 | Morgan(17.5%), the(12.0%), Charleston(9.4%), West(8.3%), Fairmont(6.4%) | the(12.9%), Morgan(11.3%), Charleston(8.8%), Fairmont(7.8%), West(7.8%) |
| vancouver-capital | Canada | Ottawa | Victoria | 6 | ✗ | +4 | -0.0129 | Victoria(33.6%), (8.5%), the(7.5%), Vancouver(5.8%), British(5.2%) | Victoria(28.5%), (11.9%), the(10.5%), called(5.6%), British(4.4%) |
| montreal-capital | Canada | Ottawa | Quebec | 11 | ✗ | +2 | -0.0510 | Quebec(27.0%), Ottawa(11.2%), the(8.7%), (7.7%), known(3.6%) | the(14.7%), Quebec(14.7%), (13.0%), Ottawa(6.1%), known(5.4%) |
| rio-de-janeiro-capital | Brazil | Brasilia | the | 4 | ✗ | -1 | +0.0186 | the(11.1%), Brazil(6.7%), (6.7%), Bra(6.7%), called(5.2%) | the(11.1%), Bra(8.6%), (7.6%), called(5.9%), known(4.6%) |
| bordeaux-capital | France | Paris | Bordeaux | 6 | ✗ | +0 | -0.0259 | Bordeaux(19.1%), Paris(14.9%), the(9.1%), (8.0%), :(4.3%) | Bordeaux(18.0%), Paris(12.4%), (9.6%), the(9.6%), called(5.2%) |
| melbourne-capital | Australia | Canberra | the | 6 | ✗ | +5 | -0.0679 | the(13.1%), (11.5%), Canberra(11.5%), known(7.0%), Melbourne(6.2%) | the(16.6%), (14.6%), :(6.9%), known(6.9%), called(6.9%) |

## Skipped

- niagara-falls-capital (a2): no_middlehop_features
- provo-capital (a2): no_middlehop_features
- glasgow-capital (a2): no_middlehop_features
- venice-capital (a2): no_middlehop_features
- bangalore-capital (a2): no_middlehop_features
