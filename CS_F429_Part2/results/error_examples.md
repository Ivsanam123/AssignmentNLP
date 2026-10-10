# Qualitative error examples (CRF, BC5CDR test)

Notation: `[entity text]_CHE` / `_DIS`. Seed 0 random sample; every example is real, none cherry-picked for effect.


## Missed entities (false negatives, no overlapping prediction) (n=1698)

- **trimethoprim** (Chemical, train count=0)
  - gold: 2 / 100 , 000 ) and those prescribed [trimethoprim]_CHE alone ( 3 .
  - pred: 2 / 100 , 000 ) and those prescribed trimethoprim alone ( 3 .
- **LOS** (Chemical, train count=0)
  - gold: Groups [ADR]_CHE ( 6 ), [ADR]_CHE + [LOS]_CHE ( 6 ) and [ADR]_CHE ( 12 ), and [ADR]_CHE + [LOS]_CHE ( 12 ) received [ADR]_CHE ( 2 mg / kg / b .
  - pred: Groups [ADR]_CHE ( 6 ), [ADR]_CHE + LOS ( 6 ) and [ADR]_CHE ( 12 ), and [ADR]_CHE + LOS ( 12 ) received [ADR]_CHE ( 2 mg / kg / b .
- **thyrotropin** (Chemical, train count=0)
  - gold: Human [corticotropin]_CHE - releasing hormone and [thyrotropin]_CHE - releasing hormone modulate the [hypercapnic]_DIS ventilatory response in humans .
  - pred: Human corticotropin - releasing hormone and thyrotropin - releasing hormone modulate the hypercapnic ventilatory response in humans .
- **diazinon** (Chemical, train count=0)
  - gold: 5 and 25mg / kg per day , intraperitoneally ) in combination with [diazinon]_CHE ( 15 mg / kg ). The levels of [cholesterol]_CHE , [triglyceride]_CHE and LDL in blood of rats were analyzed .
  - pred: 5 and 25mg / kg per day , intraperitoneally ) in combination with diazinon ( 15 mg / kg ). The levels of [cholesterol]_CHE , [triglyceride]_CHE and LDL in blood of rats were analyzed .
- **hypercalcemic** (Disease, train count=0)
  - gold: 0 mg / dL ), patients in the [hypercalcemic]_DIS group exhibited a significant increase in the mean [calcium]_CHE concentration obtained during 6 months before the switch , compared with the mean value obtained during the 7 months of observation after the switch ( 2 .
  - pred: 0 mg / dL ), patients in the hypercalcemic group exhibited a significant increase in the mean [calcium]_CHE concentration obtained during 6 months before the switch , compared with the mean value obtained during the 7 months of observation after the switch ( 2 .
- **subependymal cysts** (Disease, train count=0)
  - gold: The incidence of [subependymal cysts]_DIS in the 117 remaining infants was 14 %( 16 of 117 ). The incidence of [subependymal cysts]_DIS in infants exposed to [cocaine]_CHE prenatally was 44 %( 8 of 18 ) compared with 8 %( 8 of 99 ) in the unexposed group ( p < 0 .
  - pred: The incidence of subependymal cysts in the 117 remaining infants was 14 %( 16 of 117 ). The incidence of subependymal cysts in infants exposed to [cocaine]_CHE prenatally was 44 %( 8 of 18 ) compared with 8 %( 8 of 99 ) in the unexposed group ( p < 0 .
- **paclitaxel** (Chemical, train count=22)
  - gold: Phase 2 trial of liposomal [doxorubicin]_CHE ( 40 mg / m ( 2 )) in [platinum]_CHE / [paclitaxel]_CHE - refractory [ovarian and fallopian tube cancers]_DIS and primary [carcinoma of the peritoneum]_DIS .
  - pred: Phase 2 trial of liposomal [doxorubicin]_CHE ( 40 mg / m ( 2 )) in platinum / paclitaxel - refractory [ovarian and fallopian tube cancers]_DIS and [primary carcinoma]_DIS of the peritoneum .
- **clentiazem** (Chemical, train count=0)
  - gold: [Clentiazem]_CHE also did not prevent the enhanced responses to [calcium]_CHE seen in the [epinephrine]_CHE - treated animals , although the high dose of [clentiazem]_CHE partially attenuated the maximal response to [calcium]_CHE compared to [epinephrine]_CHE - treated animals .
  - pred: Clentiazem also did not prevent the enhanced responses to [calcium]_CHE seen in the [epinephrine]_CHE - treated animals , although the high dose of clentiazem partially attenuated the maximal response to [calcium]_CHE compared to [epinephrine]_CHE - treated animals .
- **FHF** (Disease, train count=0)
  - gold: BACKGROUND : To review evidence - based treatment options in patients with [cerebral edema]_DIS complicating [fulminant hepatic failure]_DIS ( [FHF]_DIS ) and discuss the potential applications of [hypothermia]_DIS .
  - pred: BACKGROUND : To review evidence - based treatment options in patients with [cerebral edema]_DIS complicating [fulminant hepatic failure]_DIS ( FHF ) and discuss the potential applications of [hypothermia]_DIS .
- **bradycardiac** (Disease, train count=0)
  - gold: Furthermore , selective destruction of the spinal [5-HT]_CHE nerves , produced by bilateral spinal injection of [5,7-dihydroxytryptamine]_CHE , reduced the magnitude of the vasodepressor or the [bradycardiac]_DIS responses to [clonidine]_CHE microinjected into the area near the ventrolateral surface of the medulla oblongata in rats .
  - pred: Furthermore , selective destruction of the spinal [5-HT]_CHE nerves , produced by bilateral spinal injection of [5,7-dihydroxytryptamine]_CHE , reduced the magnitude of the vasodepressor or the bradycardiac responses to [clonidine]_CHE microinjected into the area near the ventrolateral surface of the medulla oblongata in rats .

## Boundary errors (right type, wrong span) (n=584)

- **zinc acetate** (Chemical, train count=0)
  - gold: In our case the therapy with [zinc acetate]_CHE represented an effective treatment for a [Wilson's disease]_DIS patient in which [penicillamine]_CHE - related side effects appeared .
  - pred: In our case the therapy with zinc [acetate]_CHE represented an effective treatment for a [Wilson's disease]_DIS patient in which [penicillamine]_CHE - related side effects appeared .
- **depression** (Disease, train count=17)
  - gold: 53 ) or ST [depression]_DIS ( p = 0 .
  - pred: 53 ) or [ST depression]_DIS ( p = 0 .
- **edema** (Disease, train count=1)
  - gold: Fully developed [cyclophosphamide]_CHE - induced [cystitis]_DIS is characterized by nearly complete detachment of the urothelium , severe submucosal [edema]_DIS owing to damage to the microvascular bed and focal muscle [necroses]_DIS .
  - pred: Fully developed [cyclophosphamide]_CHE - induced [cystitis]_DIS is characterized by nearly complete detachment of the [urothelium]_CHE , severe [submucosal edema]_DIS owing to damage to the microvascular bed and [focal muscle necroses]_DIS .
- **chronic hepatitis B** (Disease, train count=0)
  - gold: In this study we aimed to determine the rate of YMDD motif mutants in [lamivudine]_CHE - untreated [chronic hepatitis B]_DIS patients in Iran .
  - pred: In this study we aimed to determine the rate of YMDD motif mutants in [lamivudine]_CHE - untreated chronic [hepatitis B]_DIS patients in Iran .
- **obstructive uropathy** (Disease, train count=0)
  - gold: Patients who are candidates for long - term [cyclophosphamide]_CHE treatment should be routinely evaluated for [obstructive uropathy]_DIS
  - pred: Patients who are candidates for long - term [cyclophosphamide]_CHE treatment should be routinely evaluated for obstructive [uropathy]_DIS
- **Vasurix polyvidone** (Chemical, train count=0)
  - gold: [Hexabrix]_CHE and [Vasurix polyvidone]_CHE are considered the best [contrast media]_CHE for hysterosalpingography and perhaps because of its low [toxicity]_DIS [Hexabrix]_CHE
  - pred: Hexabrix and Vasurix [polyvidone]_CHE are considered the best contrast media for [hysterosalpingography]_DIS and perhaps because of its low [toxicity]_DIS Hexabrix
- **hyper- or hypotension** (Disease, train count=0)
  - gold: Dosage groups of 15 , 30 , 50 , 75 , 100 , 125 , 150 , 175 or 200 microg [carbetocin]_CHE were assigned to blocks of three women according to the continual reassessment method ( CRM ). RESULTS : All dosage groups consisted of three women , except those with 100 microg ( n = 6 ) and 200 microg ( n = 18 ). Recorded were dose - limiting adverse events : [hyper- or hypotension]_DIS ( three ), severe [abdominal pain]_DIS ( 0 ), [vomiting]_DIS ( 0 ) and [retained placenta]_DIS ( four ). Serious adverse events occurred in seven women : six cases with [blood loss]_DIS > or = 1000 ml , four cases of manual placenta removal , five cases of additional oxytocics administration and five cases of blood transfusion .
  - pred: Dosage groups of 15 , 30 , 50 , 75 , 100 , 125 , 150 , 175 or 200 microg [carbetocin]_CHE were assigned to blocks of three women according to the continual reassessment method ( CRM ). RESULTS : All dosage groups consisted of three women , except those with 100 microg ( n = 6 ) and 200 microg ( n = 18 ). Recorded were dose - limiting adverse events : [hyper-]_DIS or [hypotension]_DIS ( three ), severe [abdominal pain]_DIS ( 0 ), [vomiting]_DIS ( 0 ) and retained placenta ( four ). Serious adverse events occurred in seven women : six cases with [blood loss]_DIS > or = 1000 ml , four cases of manual placenta removal , five cases of additional oxytocics administration and five cases of blood transfusion .
- **Renal Toxicity** (Disease, train count=2)
  - gold: Rates of [Renal Toxicity]_DIS in [Cancer]_DIS Patients Receiving [Cisplatin]_CHE With and Without [Mannitol]_CHE .
  - pred: Rates of Renal [Toxicity]_DIS in [Cancer]_DIS Patients Receiving [Cisplatin]_CHE With and Without [Mannitol]_CHE .
- **PGF2 alpha** (Chemical, train count=0)
  - gold: 05 ). [PGF2 alpha]_CHE was also significantly higher in the outer medulla of jj rats with and without [aspirin]_CHE administration ( p less than 0 .
  - pred: 05 ). [PGF2]_CHE alpha was also significantly higher in the outer medulla of jj rats with and without [aspirin]_CHE administration ( p less than 0 .
- **rapamycin** (Chemical, train count=2)
  - gold: [Sirolimus]_CHE , a mammalian target of [rapamycin]_CHE inhibitor , has been implicated as playing a special role in posttransplant [anemia]_DIS .
  - pred: [Sirolimus]_CHE , a mammalian target of [rapamycin inhibitor]_CHE , has been implicated as playing a special role in posttransplant [anemia]_DIS .

## Type errors (right span, wrong type) (n=55)

- **CM** (Chemical, train count=0)
  - gold: 014 ). This difference was not related to either a different clinical risk profile or to the volume of [CM]_CHE administered .
  - pred: 014 ). This difference was not related to either a different clinical risk profile or to the volume of [CM]_DIS administered .
- **galactose** (Chemical, train count=0)
  - gold: Beneficial effect of oral [galactose]_CHE was independent of the rat age and of the [galactose]_CHE dose ranging from 100 to 300 mg / kg / day .
  - pred: Beneficial effect of oral [galactose]_DIS was independent of the rat age and of the [galactose]_DIS dose ranging from 100 to 300 mg / kg / day .
- **polyethylene glycol-asparaginase** (Chemical, train count=0)
  - gold: [Vincristine]_CHE , [dexamethasone]_CHE , and [polyethylene glycol-asparaginase]_CHE
  - pred: [Vincristine]_CHE , [dexamethasone]_CHE , and [polyethylene glycol-asparaginase]_DIS
- **OIH** (Disease, train count=0)
  - gold: Our data suggested that the [ginsenoside Re]_CHE , but not [Rg1]_CHE or [Rb1]_CHE , may contribute toward reversal of [OIH]_DIS
  - pred: Our data suggested that the [ginsenoside]_CHE Re , but not [Rg1]_CHE or Rb1 , may contribute toward reversal of [OIH]_CHE
- **ANT-MI** (Disease, train count=0)
  - gold: Exercise - induced ST [depression]_DIS was observed in 84 % of the non - [MI]_DIS group , 43 % of the [ANT-MI]_DIS group , 38 % of the [INF-MI]_DIS group and 61 % of the total .
  - pred: Exercise - induced ST [depression]_DIS was observed in 84 % of the non - MI group , 43 % of the [ANT-MI]_CHE group , 38 % of the [INF-MI]_CHE group and 61 % of the total .
- **TNA** (Chemical, train count=0)
  - gold: Thus , the precipitating cause of [convulsions]_DIS was believed to be an [overdose]_DIS of [TNA]_CHE
  - pred: Thus , the precipitating cause of [convulsions]_DIS was believed to be an [overdose]_DIS of [TNA]_DIS
- **CM** (Chemical, train count=0)
  - gold: CONCLUSION : In this preliminary study patients submitted to PTRA showed a lower susceptibility to [renal damage]_DIS induced by [CM]_CHE administration than PCI patients .
  - pred: CONCLUSION : In this preliminary study patients submitted to PTRA showed a lower susceptibility to [renal damage]_DIS induced by [CM]_DIS administration than PCI patients .
- **XDR-TB** (Disease, train count=0)
  - gold: A 45 - year - old male patient who was on treatment with multiple second - line anti - tuberculous drugs including [linezolid]_CHE and [ethambutol]_CHE for [extensively drug-resistant tuberculosis]_DIS ( [XDR-TB]_DIS ) presented to us with painless progressive [loss of vision]_DIS in both eyes .
  - pred: A 45 - year - old male patient who was on treatment with multiple second - line anti - tuberculous drugs including linezolid and [ethambutol]_CHE for extensively [drug-resistant tuberculosis]_DIS ( [XDR-TB]_CHE ) presented to us with [painless]_DIS progressive [loss of vision]_DIS in both eyes .
- **CM** (Chemical, train count=0)
  - gold: BACKGROUND : [Renal dysfunction]_DIS induced by iodinated [contrast medium]_CHE ( [CM]_CHE ) administration can minimize the benefit of the interventional procedure in patients undergoing renal angioplasty ( PTRA ). PURPOSE : To compare the susceptibility to [nephrotoxic]_DIS effect of [CM]_CHE in patients undergoing PTRA with that of patients submitted to percutaneous coronary intervention ( PCI ). MATERIAL AND METHODS : A total of 33 patients successfully treated with PTRA ( PTRA group , mean age 70 +/- 12 years , 23 female , basal [creatinine]_CHE 1 .
  - pred: BACKGROUND : [Renal dysfunction]_DIS induced by [iodinated contrast medium]_CHE ( [CM]_DIS ) administration can minimize the benefit of the interventional procedure in patients undergoing renal angioplasty ( PTRA ). PURPOSE : To compare the susceptibility to [nephrotoxic]_DIS effect of [CM]_DIS in patients undergoing PTRA with that of patients submitted to percutaneous coronary intervention ( PCI ). MATERIAL AND METHODS : A total of 33 patients successfully treated with [PTRA]_CHE ( PTRA group , mean age 70 +/- 12 years , 23 female , basal [creatinine]_CHE 1 .
- **MP** (Chemical, train count=0)
  - gold: Careful observation of patients receiving high - dose [MP]_CHE is recommended .
  - pred: Careful observation of patients receiving high - dose [MP]_DIS is recommended .

## Spurious predictions (false positives) (n=518)

- **depressed** (Disease, train count=10)
  - gold: The patient had depressed of mental status lasting at least 24 h prior to admission .
  - pred: The patient had [depressed]_DIS of mental status lasting at least 24 h prior to admission .
- **tachyphylaxis** (Disease, train count=0)
  - gold: The purpose of the present study was to investigate whether tachyphylaxis occurs during prolonged treatment with high dose inhaled [salbutamol]_CHE .
  - pred: The purpose of the present study was to investigate whether [tachyphylaxis]_DIS occurs during prolonged treatment with high dose inhaled [salbutamol]_CHE .
- **AM149** (Chemical, train count=0)
  - gold: 001 ) occurred more frequently with AM149 than with [Disoprivan]_CHE .
  - pred: 001 ) occurred more frequently with [AM149]_CHE than with Disoprivan .
- **volume loading** (Disease, train count=0)
  - gold: Acute [heart failure]_DIS was induced by [propranolol]_CHE and volume loading after weaning from cardiopulmonary bypass ; an absence of [MR]_DIS was confirmed by echocardiography .
  - pred: Acute [heart failure]_DIS was induced by [propranolol]_CHE and [volume loading]_DIS after weaning from [cardiopulmonary bypass]_DIS ; an absence of [MR]_DIS was confirmed by echocardiography .
- **canine** (Chemical, train count=0)
  - gold: Although both dl - [sotalol]_CHE and [azimilide]_CHE rarely induced EADs in canine left ventricles , they produced frequent EADs in rabbits , in which more pronounced [QT prolongation]_DIS was seen .
  - pred: Although both dl - [sotalol]_CHE and [azimilide]_CHE rarely induced EADs in [canine]_CHE left ventricles , they produced frequent EADs in rabbits , in which more pronounced QT prolongation was seen .
- **myocardial energy** (Disease, train count=0)
  - gold: Similarly , significant disturbances in myocardial energy metabolism were detected in the [barium]_CHE - exposed rats which were consistent with the reduced contractile element shortening velocity .
  - pred: Similarly , significant disturbances in [myocardial energy]_DIS metabolism were detected in the [barium]_CHE - exposed rats which were consistent with the reduced contractile element shortening velocity .
- **liver enlargement** (Disease, train count=1)
  - gold: This study investigates the role of ILK in liver enlargement induced by [phenobarbital]_CHE ( [PB]_CHE ). Wild - type ( WT ) and ILK : liver -/- mice were given [PB]_CHE ( 0 .
  - pred: This study investigates the role of ILK in [liver enlargement]_DIS induced by [phenobarbital]_CHE ( PB ). Wild - type ( WT ) and ILK : liver -/- mice were given PB ( 0 .
- **death** (Disease, train count=26)
  - gold: Alternatively , [anemia]_DIS could result from accelerated suicidal erythrocyte death or eryptosis , which is characterized by exposure of [phosphatidylserine]_CHE ( [PS]_CHE ) at the erythrocyte surface and by cell shrinkage .
  - pred: Alternatively , [anemia]_DIS could result from accelerated suicidal erythrocyte [death]_DIS or [eryptosis]_DIS , which is characterized by exposure of [phosphatidylserine]_CHE ( PS ) at the erythrocyte surface and by cell shrinkage .
- **antidepressant** (Chemical, train count=15)
  - gold: The correlation between high serum tricyclic antidepressant concentrations and central nervous system side effects has been well established .
  - pred: The correlation between high serum tricyclic [antidepressant]_CHE concentrations and central nervous system side effects has been well established .
- **glucocorticoid** (Chemical, train count=0)
  - gold: Here , we developed a murine model of glucocorticoid - induced [glaucoma]_DIS that exhibits [glaucoma]_DIS features that are observed in patients .
  - pred: Here , we developed a murine model of [glucocorticoid]_CHE - induced [glaucoma]_DIS that exhibits [glaucoma]_DIS features that are observed in patients .
