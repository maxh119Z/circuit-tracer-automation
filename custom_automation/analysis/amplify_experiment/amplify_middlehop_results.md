# Amplify Middle-Hop Experiment Results

Variants: a2  
Candidates: 7  
Amplify factor: 2.0×  

## Summary

| Metric | Value |
|--------|-------|
| Candidates with middlehop features found | 7 / 7 |
| Baseline correct | 3 / 7 |
| Baseline wrong | 4 / 7 |
| Wrong → correct after amplification | 2 / 4 |
| Correct → still correct after amplification | 3 / 3 |
| Correct → broken by amplification | 0 / 3 |
| Flipped to middlehop intermediate (over-amplified) | 1 / 7 |
| Correct-answer rank improved (any baseline) | 4 / 7 (57.1%) |
| Mean correct-answer prob gain | +0.0157 |

## Per-Prompt Results

| Slug | Baseline | Intermediate | Correct | Predicted | # Amp | Amplified top-1 correct? | Rank Δ | Prob gain | Baseline top-5 | Amplified top-5 |
|------|----------|-------------|---------|-----------|-------|--------------------------|--------|-----------|----------------|----------------|
| mquake-63 | wrong | Islam | Muhammad | The | 6 | no | -1 | +0.0391 | The(11.6%), Islam(4.3%), Muhammad(3.8%), Prophet(3.3%), I(2.9%) | Islam(8.7%), Muhammad(7.7%), The(6.0%), Prophet(4.7%), Mohammed(2.8%) |
| mquake-855 | wrong | United Kingdom | London | The | 4 | yes | -1 | +0.0098 | London(10.5%), The(10.5%), In(3.9%), It(3.4%), I(3.0%) | London(11.5%), The(10.2%), In(3.7%), It(3.3%), I(2.6%) |
| mquake-979 | correct | Germany | German | German | 4 | yes | +0 | -0.0073 | German(13.2%), What(5.5%), English(5.5%), The(5.5%), I(4.8%) | German(12.5%), What(6.6%), The(5.9%), English(5.2%), I(5.2%) |
| mquake-542 | correct | Lithuania | Vilnius | Vilnius | 6 | yes | +0 | +0.0513 | Vilnius(8.5%), The(6.6%), Lithuania(5.9%), I(3.6%), What(3.6%) | Vilnius(13.7%), Lithuania(12.1%), Latvia(5.7%), Riga(4.4%), The(4.4%) |
| mquake-789 | wrong | Greece | Europe | Africa | 1 | yes | -1 | -0.0029 | Europe(10.2%), Africa(10.2%), The(7.0%), Asia(6.2%), I(3.3%) | Europe(9.9%), Africa(9.9%), The(6.8%), Asia(6.0%), Greece(4.1%) |
| mquake-74 | correct | Germany | German | German | 3 | yes | +0 | -0.0059 | German(10.7%), English(9.5%), I(7.4%), The(6.5%), What(5.7%) | German(10.2%), I(7.9%), The(7.0%), English(7.0%), What(6.2%) |
| mquake-581 | wrong | Islam | Muhammad | The | 3 | no | -2 | +0.0261 | The(11.3%), What(5.3%), I(4.7%), Muhammad(3.7%), A(2.2%) | The(7.1%), Muhammad(6.3%), Islam(4.9%), Al(3.8%), What(3.4%) |

