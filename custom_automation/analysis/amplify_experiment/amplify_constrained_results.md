# Amplify Experiment Results (Constrained Patching)

Variants: a2  
Candidates: 6  
Amplify factor: 5.0×  
Method: constrained patching (direct effects only — constrained_layers=range(n_layers))  

## Summary

| Metric | Value |
|--------|-------|
| Candidates with answer features found | 6 / 6 |
| Flipped to correct after amplification | 5 / 6 (83.3%) |
| Mean correct-answer prob gain | +0.1464 |

## Per-Prompt Results

| Slug | Correct | Predicted | # Amp | Flipped? | Rank Δ | Prob gain | Baseline top-5 | Amplified top-5 |
|------|---------|-----------|-------|----------|--------|-----------|----------------|----------------|
| mquake-789 | Europe | Africa | 1 | ✗ | +0 | +0.0161 | Europe(10.2%), Africa(10.2%), The(7.0%), Asia(6.2%), I(3.3%) | Africa(13.3%), Europe(11.8%), Greece(6.3%), Asia(6.3%), Australia(3.4%) |
| mquake-372 | English | What | 1 | ✓ | -1 | +0.1064 | What(8.5%), English(7.5%), I(6.6%), The(5.8%), Yes(3.1%) | English(18.2%), I(4.1%), Spanish(4.1%), What(4.1%), French(3.2%) |
| mquake-1594 | English | Italian | 2 | ✓ | -1 | +0.4385 | Italian(16.2%), English(14.4%), What(5.3%), The(5.3%), It(4.1%) | English(58.2%), Italian(5.4%), I(2.9%), American(2.0%), It(2.0%) |
| mquake-1180 | Europe | Africa | 2 | ✓ | -1 | +0.1714 | Africa(16.3%), Europe(11.2%), Asia(6.8%), The(5.3%), North(2.8%) | Europe(28.3%), Asia(15.1%), Africa(13.4%), European(3.0%), North(3.0%) |
| mquake-2465 | Europe | Africa | 3 | ✓ | -1 | +0.0430 | Africa(18.8%), Europe(14.7%), Asia(11.5%), The(4.2%), South(2.0%) | Europe(19.0%), Africa(14.8%), Asia(14.8%), America(4.8%), Eurasia(2.9%) |
| mquake-2058 | Asia | Africa | 1 | ✓ | -1 | +0.1030 | Europe(10.2%), Asia(10.2%), Africa(10.2%), North(9.0%), The(9.0%) | Asia(20.5%), Africa(18.1%), Europe(16.0%), North(7.5%), The(1.9%) |

