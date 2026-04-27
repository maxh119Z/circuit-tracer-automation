# Amplify Middle-Hop Experiment Results

Variants: a2  
Candidates: 97  
Amplify factor: 2.0×  

## Summary

| Metric | Value |
|--------|-------|
| Candidates with middlehop features found | 92 / 97 |
| Baseline correct | 71 / 92 |
| Baseline wrong | 21 / 92 |
| Wrong → correct after amplification | 1 / 21 |
| Correct → still correct after amplification | 54 / 71 |
| Correct → broken by amplification | 17 / 71 |
| Flipped to middlehop intermediate (over-amplified) | 13 / 92 |
| Correct-answer rank improved (any baseline) | 12 / 92 (13.0%) |
| Mean correct-answer prob gain | +0.0448 |

## Per-Prompt Results

| Slug | Baseline | Intermediate | Correct | Predicted | # Amp | Amplified top-1 correct? | Rank Δ | Prob gain | Baseline top-5 | Amplified top-5 |
|------|----------|-------------|---------|-----------|-------|--------------------------|--------|-----------|----------------|----------------|
| birmingham-capital | wrong | Alabama | Montgomery | the | 3 | no | -8 | +0.0190 | the(18.5%), Birmingham(9.9%), (8.7%), called(5.3%), located(5.3%) | Birmingham(32.4%), Alabama(25.2%), the(6.4%), (3.4%), Montgomery(3.4%) |
| anchorage-capital | correct | Alaska | Juneau | Juneau | 5 | yes | +0 | +0.0059 | Juneau(30.9%), Anchorage(16.5%), the(11.4%), (6.1%), Alaska(4.2%) | Juneau(31.4%), Anchorage(27.7%), Alaska(24.5%), Fairbanks(6.2%), the(0.8%) |
| grand-canyon-capital | correct | Arizona | Phoenix | Phoenix | 6 | yes | +0 | +0.0039 | Phoenix(28.3%), Flagstaff(7.2%), Prescott(7.2%), (6.3%), :(5.6%) | Phoenix(28.7%), Flagstaff(13.6%), Arizona(10.5%), Prescott(9.3%), Tucson(6.4%) |
| fort-smith-capital | wrong | Arkansas | Little Rock | Fort | 3 | no | -1 | +0.0264 | Fort(13.0%), the(10.2%), (8.9%), Little(7.9%), Fayetteville(7.0%) | Arkansas(22.4%), Fayetteville(15.3%), Little(10.5%), Fort(9.3%), the(5.0%) |
| los-angeles-capital | correct | California | Sacramento | Sacramento | 12 | yes | +0 | +0.0410 | Sacramento(28.5%), the(11.9%), (7.2%), California(7.2%), called(5.6%) | Sacramento(32.6%), California(25.4%), Los(8.3%), San(5.7%), Santa(3.4%) |
| new-haven-capital | correct | Connecticut | Hartford | Hartford | 4 | yes | +0 | +0.0625 | Hartford(33.6%), the(10.9%), Connecticut(6.6%), (5.8%), New(4.5%) | Hartford(39.8%), Connecticut(24.1%), Bridgeport(6.9%), the(2.9%), CT(2.9%) |
| wilmington-capital | wrong | Delaware | Dover | Raleigh | 2 | no | -1 | +0.0239 | Raleigh(20.2%), the(9.5%), Dover(8.4%), Wilmington(6.5%), (5.8%) | Raleigh(17.8%), Wilmington(10.8%), Dover(10.8%), Delaware(8.4%), the(7.4%) |
| miami-capital | correct | Florida | Tallahassee | Tallahassee | 7 | yes | +0 | -0.0996 | Tallahassee(43.9%), the(7.6%), Miami(6.7%), (5.9%), Florida(4.1%) | Tallahassee(34.0%), Florida(20.6%), Miami(12.5%), Jacksonville(5.2%), Tampa(4.1%) |
| savannah-capital | correct | Georgia | Atlanta | Atlanta | 6 | yes | +0 | +0.1602 | Atlanta(14.5%), Augusta(11.3%), the(11.3%), Savannah(7.8%), (6.8%) | Atlanta(30.5%), Georgia(23.7%), Savannah(12.7%), Augusta(9.9%), Macon(6.8%) |
| pearl-harbor-capital | correct | Hawaii | Honolulu | Honolulu | 4 | yes | +0 | +0.0156 | Honolulu(52.0%), Hawaii(7.0%), :(4.9%), the(4.3%), (3.8%) | Honolulu(53.5%), Hawaii(13.6%), :(3.4%), Oahu(3.4%), (2.7%) |
| twin-falls-capital | correct | Idaho | Boise | Boise | 1 | no | +1 | -0.0459 | Boise(20.0%), Twin(12.2%), the(10.7%), (7.4%), Idaho(7.4%) | Twin(17.5%), Boise(15.4%), the(9.3%), Idaho(8.3%), (6.4%) |
| chicago-capital | wrong | Illinois | Springfield | Illinois | 1 | no | -1 | +0.0322 | Illinois(25.8%), Springfield(13.8%), the(12.2%), (7.4%), called(4.0%) | the(17.0%), Springfield(17.0%), (10.3%), called(6.2%), :(4.9%) |
| south-bend-capital | correct | Indiana | Indianapolis | Indianapolis | 3 | yes | +0 | +0.2314 | Indianapolis(21.8%), the(9.1%), South(7.1%), (6.2%), Indiana(6.2%) | Indianapolis(44.9%), Indiana(30.9%), Fort(1.5%), the(1.4%), South(1.4%) |
| cedar-rapids-capital | wrong | Iowa | Des Moines | Iowa | 2 | no | +0 | +0.0898 | Iowa(33.2%), Des(22.9%), the(5.8%), (5.1%), Cedar(4.0%) | Iowa(46.5%), Des(31.8%), Cedar(3.8%), Ames(2.0%), Dubuque(1.6%) |
| wichita-capital | correct | Kansas | Topeka | Topeka | 3 | no | +1 | +0.0381 | Topeka(19.5%), the(11.8%), Kansas(11.8%), Wichita(9.2%), (6.3%) | Kansas(43.6%), Topeka(23.3%), Wichita(14.2%), the(1.9%), KS(1.5%) |
| louisville-capital | correct | Kentucky | Frankfort | Frankfort | 3 | yes | +0 | -0.0352 | Frankfort(32.4%), the(12.0%), Kentucky(9.3%), (8.2%), Louisville(8.2%) | Frankfort(28.9%), Kentucky(28.9%), Louisville(15.4%), Lexington(7.3%), the(4.4%) |
| bar-harbor-capital | correct | Maine | Augusta | Augusta | 6 | no | +1 | -0.1494 | Augusta(33.2%), Bangor(15.7%), the(9.5%), (7.4%), Ellsworth(3.5%) | Bangor(38.7%), Augusta(18.3%), Maine(18.3%), Portland(5.2%), Ellsworth(3.6%) |
| baltimore-capital | correct | Maryland | Annapolis | Annapolis | 2 | yes | +0 | -0.0312 | Annapolis(44.1%), the(8.7%), Baltimore(6.0%), (6.0%), Maryland(5.2%) | Annapolis(41.0%), Maryland(24.9%), Baltimore(22.0%), the(2.0%), (1.2%) |
| cambridge-capital | correct | Massachusetts | Boston | Boston | 6 | yes | +0 | +0.1094 | Boston(26.2%), the(10.9%), (8.5%), Massachusetts(4.0%), Cambridge(4.0%) | Boston(37.1%), Massachusetts(12.1%), the(6.4%), (5.7%), Cambridge(4.4%) |
| detroit-capital | correct | Michigan | Lansing | Lansing | 7 | yes | +0 | -0.1055 | Lansing(53.1%), the(7.2%), Michigan(7.2%), (5.6%), called(3.8%) | Lansing(42.6%), Michigan(29.3%), Detroit(8.4%), Ann(1.9%), Flint(1.6%) |
| minneapolis-capital | wrong | Minnesota | Saint Paul | St | 4 | no | +2 | -0.0688 | St(24.3%), the(14.7%), Saint(14.7%), Minnesota(10.2%), Minneapolis(5.4%) | Minnesota(35.4%), Minneapolis(21.4%), St(10.1%), Saint(7.9%), MN(6.1%) |
| biloxi-capital | correct | Mississippi | Jackson | Jackson | 3 | yes | +0 | +0.0713 | Jackson(17.6%), the(15.5%), Bil(6.4%), Gulf(5.7%), (5.0%) | Jackson(24.7%), the(10.3%), Mississippi(8.0%), Bil(8.0%), Gulf(7.1%) |
| st-louis-capital | correct | Missouri | Jefferson City | Jefferson | 4 | no | +1 | -0.0771 | Jefferson(25.4%), the(10.5%), (10.5%), called(6.4%), St(5.7%) | Missouri(29.1%), Jefferson(17.7%), St(15.6%), MO(4.0%), Columbia(4.0%) |
| billings-capital | correct | Montana | Helena | Helena | 10 | yes | +0 | +0.1953 | Helena(35.2%), the(16.6%), (7.9%), :(3.7%), Bo(3.3%) | Helena(54.7%), Montana(13.8%), Billings(5.8%), Bo(5.1%), the(2.4%) |
| omaha-capital | correct | Nebraska | Lincoln | Lincoln | 3 | yes | +0 | -0.0020 | Lincoln(45.1%), the(8.9%), Nebraska(7.8%), (6.1%), Omaha(4.2%) | Lincoln(44.9%), Nebraska(24.0%), Omaha(14.6%), the(2.5%), called(1.5%) |
| las-vegas-capital | wrong | Nevada | Carson City | Reno | 2 | no | +1 | -0.0078 | Reno(17.5%), Nevada(12.0%), (9.4%), Carson(9.4%), the(9.4%) | Nevada(30.1%), Reno(18.2%), Las(12.5%), Carson(8.6%), (5.2%) |
| nashua-capital | correct | New Hampshire | Concord | Concord | 2 | yes | +0 | +0.0137 | Concord(40.2%), Manchester(14.8%), the(7.9%), Nash(4.8%), New(3.7%) | Concord(41.6%), Manchester(15.3%), Nash(8.2%), New(7.2%), the(3.4%) |
| atlantic-city-capital | wrong | New Jersey | Trenton | Atlantic | 10 | no | +1 | +0.0273 | Atlantic(17.6%), Trenton(15.5%), (9.4%), the(7.3%), New(3.9%) | New(26.6%), Atlantic(20.6%), Trenton(18.3%), Camden(4.6%), Ocean(1.9%) |
| albuquerque-capital | correct | New Mexico | Santa Fe | Santa | 2 | no | +0 | +0.0332 | Santa(36.9%), the(10.6%), New(9.3%), Albuquerque(6.4%), (5.7%) | Santa(40.2%), New(18.9%), Albuquerque(7.9%), the(3.7%), Las(2.9%) |
| charlotte-capital | correct | North Carolina | Raleigh | Raleigh | 5 | yes | +0 | +0.0527 | Raleigh(37.3%), North(9.5%), the(8.3%), (5.1%), Charlotte(4.5%) | Raleigh(42.6%), North(13.8%), Charlotte(7.4%), NC(4.0%), the(3.5%) |
| fargo-capital | correct | North Dakota | Bismarck | Bismarck | 3 | yes | +0 | +0.1289 | Bismarck(25.4%), North(17.4%), Fargo(15.3%), the(9.3%), (5.0%) | Bismarck(38.3%), Fargo(29.7%), North(8.5%), the(2.8%), Moor(2.4%) |
| cleveland-capital | correct | Ohio | Columbus | Columbus | 2 | yes | +0 | +0.0352 | Columbus(31.6%), the(10.3%), Cleveland(7.1%), (6.2%), Ohio(5.5%) | Columbus(35.2%), Ohio(11.4%), Cleveland(11.4%), Akron(6.1%), the(6.1%) |
| tulsa-capital | correct | Oklahoma | Oklahoma City | Oklahoma | 4 | no | +0 | +0.2246 | Oklahoma(41.2%), the(10.4%), (7.2%), :(3.4%), Tulsa(3.0%) | Oklahoma(63.7%), OK(8.6%), Tulsa(7.6%), the(2.5%), Ok(1.7%) |
| portland-or-capital | correct | Oregon | Salem | Salem | 2 | yes | +0 | +0.0342 | Salem(21.2%), the(12.9%), (8.8%), Portland(6.9%), Augusta(6.1%) | Salem(24.6%), the(14.9%), (7.1%), Portland(4.9%), Augusta(3.8%) |
| philadelphia-capital | correct | Pennsylvania | Harrisburg | Harrisburg | 3 | yes | +0 | +0.0527 | Harrisburg(46.7%), the(10.4%), (5.6%), Pennsylvania(5.6%), called(3.8%) | Harrisburg(52.0%), the(9.0%), Pennsylvania(5.5%), (4.8%), Philadelphia(3.8%) |
| newport-capital | wrong | Rhode Island | Providence | the | 2 | no | -2 | +0.0447 | the(15.0%), Newport(9.1%), (6.2%), Providence(3.0%), Harrisburg(3.0%) | Newport(15.6%), the(12.2%), Providence(7.4%), Rhode(5.8%), (4.5%) |
| myrtle-beach-capital | correct | South Carolina | Columbia | Columbia | 7 | no | +0 | +0.0156 | Columbia(23.6%), Myrtle(20.8%), Conway(6.0%), the(6.0%), Florence(5.3%) | Myrtle(25.2%), Columbia(25.2%), Charleston(7.2%), Conway(6.4%), South(6.4%) |
| mount-rushmore-capital | wrong | South Dakota | Pierre | South | 2 | no | -3 | +0.0625 | South(12.8%), :(8.8%), (7.8%), the(7.8%), (6.8%) | South(16.8%), Pierre(13.1%), Sioux(11.5%), Rapid(5.4%), the(4.2%) |
| memphis-capital | wrong | Tennessee | Nashville | the | 1 | no | +5 | -0.1160 | the(16.2%), Nashville(14.3%), (11.1%), Memphis(9.8%), Tennessee(8.7%) | the(25.2%), (19.6%), Jackson(6.4%), :(4.4%), ...(3.0%) |
| dallas-capital | correct | Texas | Austin | Austin | 9 | yes | +0 | +0.0566 | Austin(34.2%), the(11.1%), (7.6%), Fort(4.1%), Texas(4.1%) | Austin(39.8%), Texas(14.6%), Dallas(7.9%), Fort(6.1%), Houston(3.3%) |
| provo-capital | wrong | Utah | Salt Lake City | Provo | 5 | no | -1 | +0.0713 | Provo(21.9%), Salt(19.2%), the(9.1%), Utah(8.1%), (3.3%) | Salt(26.4%), Utah(23.2%), Provo(20.5%), Brigham(6.6%), Ogden(4.0%) |
| burlington-vt-capital | wrong | Vermont | Montpelier | the | 3 | no | -2 | +0.0295 | the(12.3%), (6.6%), Mont(5.8%), Madison(5.1%), Burlington(4.5%) | the(8.8%), Burlington(8.8%), Mont(8.8%), Vermont(5.3%), Madison(4.7%) |
| virginia-beach-capital | correct | Virginia | Richmond | Richmond | 1 | no | +1 | +0.0322 | Richmond(22.8%), Norfolk(17.7%), the(12.2%), (7.4%), Hampton(4.0%) | Virginia(37.7%), Richmond(26.0%), Norfolk(9.5%), Hampton(4.5%), Williamsburg(4.0%) |
| seattle-capital | correct | Washington | Olympia | Olympia | 6 | yes | +0 | -0.0781 | Olympia(60.2%), the(7.2%), (4.9%), Washington(4.4%), called(2.1%) | Olympia(52.3%), Seattle(15.0%), Tacoma(8.0%), Washington(2.3%), the(2.3%) |
| morgantown-capital | wrong | West Virginia | Charleston | Morgan | 5 | no | -1 | +0.0142 | Morgan(17.5%), the(12.0%), Charleston(9.4%), West(8.3%), Fairmont(6.4%) | Morgan(17.8%), Charleston(10.8%), Fairmont(9.5%), the(9.5%), (5.8%) |
| milwaukee-capital | correct | Wisconsin | Madison | Madison | 4 | no | +1 | +0.0176 | Madison(22.5%), the(12.0%), Milwaukee(10.6%), Wisconsin(10.6%), (5.7%) | Wisconsin(31.1%), Madison(24.2%), Milwaukee(18.8%), Green(4.2%), the(2.9%) |
| toronto-capital | correct | Canada | Ottawa | Ottawa | 14 | yes | +0 | +0.0439 | Ottawa(23.3%), Toronto(9.7%), the(9.7%), (7.6%), Ontario(5.2%) | Ottawa(27.7%), Toronto(19.0%), Ontario(14.8%), Canada(10.2%), Montreal(4.2%) |
| vancouver-capital | wrong | Canada | Ottawa | Victoria | 6 | no | -7 | +0.0581 | Victoria(33.6%), (8.5%), the(7.5%), Vancouver(5.8%), British(5.2%) | Victoria(24.4%), Vancouver(14.8%), Ottawa(8.0%), Toronto(6.2%), Canada(6.2%) |
| montreal-capital | wrong | Canada | Ottawa | Quebec | 15 | no | +0 | +0.0977 | Quebec(27.0%), Ottawa(11.2%), the(8.7%), (7.7%), known(3.6%) | Quebec(30.7%), Ottawa(21.0%), Toronto(7.7%), Canada(6.8%), Montreal(6.0%) |
| guadalajara-capital | correct | Mexico | Mexico City | Mexico | 7 | no | +0 | +0.1641 | Mexico(30.5%), the(11.2%), Guadalajara(9.9%), (6.0%), called(4.1%) | Mexico(46.9%), Guadalajara(13.5%), Mexican(4.4%), Jalisco(3.9%), México(3.4%) |
| cancun-capital | correct | Mexico | Mexico City | Mexico | 6 | no | +0 | +0.2891 | Mexico(32.0%), the(11.8%), (7.2%), called(6.3%), a(3.0%) | Mexico(60.9%), Mexican(3.0%), Cancun(3.0%), Guadalajara(2.7%), Monterrey(2.1%) |
| rio-de-janeiro-capital | wrong | Brazil | Brasilia | the | 2 | no | -1 | +0.0225 | the(11.1%), Brazil(6.7%), (6.7%), Bra(6.7%), called(5.2%) | the(11.5%), Bra(9.0%), (7.9%), called(5.4%), :(4.8%) |
| sao-paulo-capital | correct | Brazil | Brasilia | Bra | 4 | no | +0 | +0.1123 | Bra(13.2%), the(10.3%), São(8.0%), Sao(7.1%), Rio(6.2%) | Bra(24.4%), Rio(7.9%), the(7.0%), Sao(6.2%), São(4.2%) |
| medellin-capital | correct | Colombia | Bogota | Bogota | 1 | yes | +0 | +0.0938 | Bogota(22.3%), Bogotá(10.5%), called(8.2%), the(6.4%), Colombia(5.0%) | Bogota(31.6%), Bogotá(14.9%), Colombia(10.3%), Medellín(6.2%), called(3.8%) |
| cusco-capital | correct | Peru | Lima | Lima | 1 | yes | +0 | +0.0527 | Lima(34.6%), the(14.4%), (6.0%), located(5.3%), called(4.1%) | Lima(39.8%), the(13.0%), located(4.8%), (4.8%), Peru(4.2%) |
| cork-capital | correct | Ireland | Dublin | Dublin | 14 | yes | +0 | +0.1113 | Dublin(31.6%), the(10.3%), (9.0%), Cork(6.2%), called(5.5%) | Dublin(42.8%), Cork(13.9%), Ireland(8.4%), Limerick(7.4%), Irish(3.1%) |
| marseille-capital | correct | France | Paris | Paris | 5 | yes | +0 | +0.1328 | Paris(19.5%), the(9.2%), (8.1%), Lyon(7.2%), Marseille(4.3%) | Paris(32.8%), France(17.6%), Lyon(6.5%), Marseille(5.7%), French(5.7%) |
| lyon-capital | correct | France | Paris | Paris | 5 | yes | +0 | +0.1885 | Paris(18.8%), the(10.1%), Lyon(10.1%), (8.9%), :(4.8%) | Paris(37.7%), France(20.2%), Lyon(9.6%), French(5.8%), the(2.1%) |
| bordeaux-capital | wrong | France | Paris | Bordeaux | 5 | yes | -1 | +0.1670 | Bordeaux(19.1%), Paris(14.9%), the(9.1%), (8.0%), :(4.3%) | Paris(31.6%), Bordeaux(19.2%), France(14.9%), Toulouse(6.2%), French(3.8%) |
| munich-capital | correct | Germany | Berlin | Berlin | 13 | no | +1 | -0.0166 | Berlin(25.2%), Munich(15.2%), the(8.2%), (6.3%), Germany(3.9%) | Munich(23.5%), Berlin(23.5%), Bavaria(8.6%), Germany(6.7%), the(3.6%) |
| hamburg-capital | correct | Germany | Berlin | Berlin | 12 | yes | +0 | +0.1191 | Berlin(29.5%), (8.4%), Hamburg(7.5%), the(6.6%), :(4.5%) | Berlin(41.4%), Hamburg(11.9%), Germany(7.2%), Bremen(3.9%), Bonn(2.1%) |
| frankfurt-capital | correct | Germany | Berlin | Berlin | 10 | no | +1 | +0.0518 | Berlin(13.9%), Frankfurt(12.3%), the(8.4%), (7.4%), Germany(6.6%) | Germany(21.6%), Berlin(19.0%), Frankfurt(13.1%), German(4.8%), Wiesbaden(4.2%) |
| milan-capital | correct | Italy | Rome | Rome | 7 | yes | +0 | +0.0449 | Rome(18.4%), the(11.1%), (8.7%), Milan(6.8%), called(5.3%) | Rome(22.9%), Milan(12.3%), Italy(9.5%), the(8.4%), (4.5%) |
| venice-capital | wrong | Italy | Rome | the | 3 | no | +3 | -0.0291 | Rome(9.1%), the(9.1%), (8.0%), Venice(8.0%), :(6.2%) | the(10.2%), (9.0%), :(7.0%), called(7.0%), Rome(6.2%) |
| florence-capital | correct | Italy | Rome | Rome | 7 | yes | +0 | +0.0977 | Rome(24.6%), the(11.6%), (8.0%), Florence(8.0%), :(4.3%) | Rome(34.4%), Florence(9.9%), the(8.7%), (4.7%), Italy(3.6%) |
| naples-capital | correct | Italy | Rome | Rome | 7 | yes | +0 | +0.0859 | Rome(26.8%), the(9.8%), (7.7%), called(6.0%), :(4.1%) | Rome(35.4%), Italy(10.2%), Naples(6.2%), Milan(4.8%), the(4.2%) |
| barcelona-capital | correct | Spain | Madrid | Madrid | 2 | yes | +0 | -0.1279 | Madrid(27.3%), (8.9%), the(8.9%), Barcelona(4.7%), called(4.7%) | Madrid(14.6%), (11.3%), the(11.3%), called(7.8%), :(6.1%) |
| seville-capital | correct | Spain | Madrid | Madrid | 4 | yes | +0 | -0.0684 | Madrid(41.8%), the(9.3%), (5.7%), Seville(3.9%), known(2.1%) | Madrid(35.0%), Seville(7.8%), the(7.8%), (5.3%), Sevilla(2.9%) |
| valencia-capital | correct | Spain | Madrid | Madrid | 5 | yes | +0 | -0.0332 | Madrid(25.8%), Valencia(13.8%), the(10.7%), (6.5%), called(3.5%) | Madrid(22.5%), Valencia(17.5%), the(7.3%), (5.0%), Alicante(3.4%) |
| porto-capital | correct | Portugal | Lisbon | Lisbon | 3 | yes | +0 | +0.0977 | Lisbon(34.2%), Porto(11.1%), the(8.6%), (5.2%), a(3.6%) | Lisbon(43.9%), Porto(12.6%), Portugal(7.6%), the(3.6%), Lisboa(3.2%) |
| rotterdam-capital | correct | Netherlands | Amsterdam | Amsterdam | 9 | yes | +0 | +0.1328 | Amsterdam(16.6%), The(12.9%), (10.1%), the(8.9%), Rotterdam(5.4%) | Amsterdam(29.9%), Rotterdam(9.7%), Utrecht(6.7%), Holland(6.7%), Netherlands(5.9%) |
| antwerp-capital | correct | Belgium | Brussels | Brussels | 4 | yes | +0 | +0.1172 | Brussels(37.9%), (10.8%), the(7.5%), Belgium(5.8%), :(3.1%) | Brussels(49.6%), (7.6%), the(5.9%), Belgium(4.6%), :(1.9%) |
| salzburg-capital | correct | Austria | Vienna | Vienna | 4 | yes | +0 | +0.0898 | Vienna(30.5%), the(9.9%), (7.7%), Austria(6.8%), Salzburg(5.3%) | Vienna(39.5%), Austria(21.1%), Salzburg(10.0%), Linz(3.2%), the(2.9%) |
| gothenburg-capital | correct | Sweden | Stockholm | Stockholm | 11 | yes | +0 | +0.0391 | Stockholm(26.4%), Gothenburg(11.0%), the(9.7%), (8.6%), Sweden(5.2%) | Stockholm(30.3%), Sweden(23.5%), Gothenburg(14.3%), Swedish(5.9%), Göteborg(5.2%) |
| bergen-capital | correct | Norway | Oslo | Oslo | 3 | yes | +0 | +0.0137 | Oslo(39.8%), Bergen(12.9%), the(6.9%), (6.1%), :(3.3%) | Oslo(41.2%), Bergen(13.4%), the(5.6%), (4.9%), Stavanger(2.6%) |
| krakow-capital | correct | Poland | Warsaw | Warsaw | 3 | yes | +0 | +0.0898 | Warsaw(22.7%), the(13.8%), Poland(8.3%), (5.7%), a(4.5%) | Warsaw(31.6%), Poland(14.9%), the(8.0%), Krakow(4.9%), (3.3%) |
| thessaloniki-capital | correct | Greece | Athens | Athens | 6 | yes | +0 | +0.2266 | Athens(16.8%), the(14.8%), (9.0%), a(5.5%), also(4.2%) | Athens(39.5%), Thessaloniki(14.5%), Greece(11.3%), the(4.7%), Greek(2.2%) |
| st-petersburg-capital | correct | Russia | Moscow | Moscow | 11 | yes | +0 | +0.1611 | Moscow(20.8%), the(9.9%), called(7.7%), (6.8%), Leningrad(6.0%) | Moscow(36.9%), Russia(22.5%), Leningrad(12.0%), Russian(5.0%), St(3.0%) |
| istanbul-capital | correct | Turkey | Ankara | Ankara | 6 | yes | +0 | +0.0449 | Ankara(34.2%), the(12.5%), Turkey(6.7%), (4.6%), Istanbul(4.1%) | Ankara(38.7%), Turkey(23.4%), Istanbul(16.1%), Turkish(4.1%), the(1.9%) |
| shanghai-capital | correct | China | Beijing | Beijing | 10 | yes | +0 | +0.0176 | Beijing(35.0%), (6.9%), the(6.1%), :(3.7%), called(3.2%) | Beijing(36.7%), China(15.3%), Shanghai(9.3%), Nanjing(5.0%), Hangzhou(4.4%) |
| osaka-capital | correct | Japan | Tokyo | Tokyo | 11 | yes | +0 | +0.0332 | Tokyo(27.3%), Kyoto(14.6%), Osaka(11.4%), (6.1%), the(4.7%) | Tokyo(30.7%), Kyoto(16.4%), Osaka(12.8%), Japan(12.8%), Nagoya(2.8%) |
| kyoto-capital | correct | Japan | Tokyo | Tokyo | 10 | yes | +0 | +0.0547 | Tokyo(25.4%), Osaka(10.6%), the(8.3%), (5.7%), Kyoto(4.4%) | Tokyo(30.9%), Japan(16.5%), Osaka(14.6%), Kyoto(10.0%), Japanese(4.7%) |
| busan-capital | correct | South Korea | Seoul | Seoul | 1 | yes | +0 | -0.0586 | Seoul(30.1%), (6.7%), Dae(6.7%), the(4.6%), South(3.6%) | Seoul(24.2%), South(14.6%), Dae(6.1%), (6.1%), the(4.2%) |
| mumbai-capital | correct | India | New Delhi | Delhi | 6 | no | +1 | -0.0645 | Delhi(20.1%), New(13.9%), the(9.5%), (5.8%), Mumbai(5.8%) | Delhi(29.3%), Mumbai(10.8%), New(7.4%), Maharashtra(6.5%), India(5.1%) |
| bangalore-capital | wrong | India | New Delhi | Karnataka | 5 | no | +1 | -0.0002 | Bengaluru(11.8%), Karnataka(11.8%), Bangalore(11.8%), the(10.4%), (5.5%) | Bangalore(11.6%), Bengaluru(11.6%), Karnataka(11.6%), the(10.3%), (5.5%) |
| karachi-capital | correct | Pakistan | Islamabad | Islamabad | 4 | yes | +0 | +0.0508 | Islamabad(37.9%), Pakistan(6.6%), Karachi(6.6%), Lahore(5.1%), the(5.1%) | Islamabad(43.0%), Pakistan(10.9%), Karachi(9.6%), Lahore(6.6%), the(3.1%) |
| phuket-capital | correct | Thailand | Bangkok | Bangkok | 5 | yes | +0 | +0.1289 | Bangkok(66.0%), the(5.4%), (4.2%), called(1.6%), a(1.4%) | Bangkok(78.9%), Thailand(3.5%), Phuket(2.1%), Chiang(1.9%), bangkok(1.4%) |
| ho-chi-minh-capital | correct | Vietnam | Hanoi | Hanoi | 11 | yes | +0 | -0.0215 | Hanoi(44.3%), the(8.7%), Saigon(6.0%), (5.3%), Vietnam(4.1%) | Hanoi(42.2%), Vietnam(22.6%), Saigon(15.5%), Vietnamese(2.7%), Viet(1.6%) |
| dubai-capital | correct | United Arab Emirates | Abu Dhabi | Abu | 7 | no | +0 | -0.0273 | Abu(47.5%), the(8.3%), (5.7%), Dubai(4.4%), called(3.9%) | Abu(44.7%), Dubai(18.7%), UAE(16.5%), Sharjah(4.2%), Emir(2.9%) |
| alexandria-capital | correct | Egypt | Cairo | Cairo | 5 | yes | +0 | +0.0098 | Cairo(34.6%), the(8.7%), (5.3%), Alexandria(5.3%), Egypt(3.2%) | Cairo(35.5%), Alexandria(7.9%), the(7.9%), Egypt(5.4%), (4.2%) |
| sydney-capital | correct | Australia | Canberra | Canberra | 8 | no | +1 | +0.0020 | Canberra(21.2%), New(11.3%), (8.8%), the(8.8%), called(5.3%) | Australia(24.2%), Canberra(21.4%), Sydney(13.0%), Melbourne(8.9%), Australian(8.9%) |
| melbourne-capital | wrong | Australia | Canberra | the | 8 | no | +0 | +0.0742 | the(13.1%), (11.5%), Canberra(11.5%), known(7.0%), Melbourne(6.2%) | Melbourne(27.5%), Canberra(18.9%), Australia(11.5%), Victoria(6.9%), the(4.2%) |

## Skipped

- new-orleans-capital (a2): no_middlehop_features
- niagara-falls-capital (a2): no_middlehop_features
- manchester-capital (a2): no_middlehop_features
- liverpool-capital (a2): no_middlehop_features
- glasgow-capital (a2): no_middlehop_features
