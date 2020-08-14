# High-Frequency-Credit-Rating

UBS Shadow Rating Project - This project use high frequency inputs including implied probability of default from CDS and Bond to predict credit rating on company level

## CDS Pricing
For each company, we have multiple maturities for CDS contract and this model implies piecewise constant hazard rate to get the curve of Survival Probability along time. The mathmatical model behind calculation of each function in the code (CDSPricer.py) is as below:

### get_discount_factor(self, yieldcurveTenor, yieldcurveRate, t)
<img src="https://render.githubusercontent.com/render/math?math=Discount Factor = e^{-rt}">

### getSurvivalProbability(self, creditcurveTenor, creditcurveSP, t)
Given a credit curve which shows the survival probability changes along time, the goal of this function is to bootstrap the Survival Prob inside a time partition.<br/>
<img src="https://render.githubusercontent.com/render/math?math=ProbSurvival_{T} = ProbSurvival_{t}*e^{-hazard*(T-t)}">

### calculatePremiumLeg(self, creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, cdsMaturity, num_premium_year, accruedPremiumFlag, spread, h):
This is to get the sum of accumulated accrued and premium leg. <br/>
<img src="https://render.githubusercontent.com/render/math?math=Premium = spread * (annuity %2B accruedPremium)">

### calculateDefaultLeg(self, creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, cdsMaturity, num_default_year, recoveryRate, h)
The goal is to calculate accumulated loss given the prob of Survival Curve

### bootstrapCDSspread(self)
This funtion use Newton's method to find the hazard rate curve given credit spreads curve
