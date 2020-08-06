import numpy as np
import pandas as pd
from scipy.optimize import newton

class cds:
    def __init__(self, creditcurve, creditcurveLength, yieldcurve, yieldcurveLength,
                 cdsTenor, premiumFrequency, defaultFrequency, accruedPremium, recoveryRate, spread):
        self.creditcurve = creditcurve
        self.creditcurveLength = creditcurveLength
        self.yieldcurve = yieldcurve
        self.yieldcurveLength = yieldcurveLength
        self.cdsTenor = cdsTenor
        self.premiumFrequency = premiumFrequency
        self.defaultFrequency = defaultFrequency
        self.accruedPremium = accruedPremium
        self.recoveryRate = recoveryRate
        self.spread = spread
    
    def get_discount_factor(self, yieldcurveTenor, yieldcurveRate, t):
        result = -1
        min_time_index = 0
        max_time_index = len(yieldcurveTenor) - 1
        if t < 0:
            result = - 1
        elif t == 0:
            result = 1.0
        elif t > 0 and t <= yieldcurveTenor[min_time_index]:
            result = np.exp(-t*yieldcurveRate[0])
        elif t >= yieldcurveTenor[max_time_index]:
            result = np.exp(-t*yieldcurveRate[-1])
        else:
            for i in range(max_time_index):
                if t >= yieldcurveTenor[i] and t < yieldcurveTenor[i+1]:
                    yield_interpolated = yieldcurveRate[i] + (yieldcurveRate[i+1] - yieldcurveRate[i]) / \
                        (yieldcurveTenor[i+1]-yieldcurveTenor[i]) * (t-yieldcurveTenor[i])
                    result = np.exp(-t*yield_interpolated)
        return result
    
    # Credit Curve is the Survival Probability Curve
    def getSurvivalProbability(self, creditcurveTenor, creditcurveSP, t):
        result = -1
        min_time_index = 0
        max_time_index = len(creditcurveTenor) - 1
        if t < 0:
            result = -1
        elif t == 0:
            result = 1
        elif t > 0 and t <= creditcurveTenor[min_time_index]:
            h = -np.log(creditcurveSP[0] / creditcurveTenor[min_time_index])
            result = np.exp(-h*t)
        elif t == creditcurveTenor[max_time_index]:
            result = creditcurveSP[-1]
        elif t > creditcurveTenor[max_time_index]:
            h = 0
            if len(creditcurveTenor) == 1:
                h = - np.log(creditcurveSP[-1]) / creditcurveTenor[max_time_index]
            else: 
                h = - np.log(creditcurveSP[-1]/creditcurveSP[-2]) / \
                        (creditcurveTenor[-1]-creditcurveTenor[-2])
                result = creditcurveSP[-1] * np.exp(-(t - creditcurveTenor[max_time_index])*h)
        else:  # where t is in between min_time and max_time
            for i in range(max_time_index):
                if t >= creditcurveTenor[i] and t < creditcurveTenor[i+1]:
                    # h is the piecewise hazard rate
                    h = -np.log(creditcurveSP[i+1]/creditcurveSP[i]) / \
                        (creditcurveTenor[i+1]-creditcurveTenor[i])
#                     if h > 1:
#                         h = 1
                    result = creditcurveSP[i] * \
                        np.exp(-(t-creditcurveTenor[i])*h)
        return result
    
    def calculatePremiumLeg(self, creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, cdsMaturity,\
                        num_premium_year, accruedPremiumFlag, spread, h):
        max_time_index = len(creditcurveTenor) - 1
        if max_time_index > 0 and cdsMaturity <= creditcurveTenor[max_time_index]:
                annuity = 0
                accruedPremium = 0
                N = int(cdsMaturity*num_premium_year)
                for n in range(1, N+1):
                    tn = n / num_premium_year
                    tnm1 = (n-1) / num_premium_year
                    dt = 1.0 / num_premium_year
                    annuity += dt * \
                        self.get_discount_factor(yieldcurveTenor, yieldcurveRate, tn)*self.getSurvivalProbability(creditcurveTenor,creditcurveSP, tn)
                    if accruedPremiumFlag:
                        accruedPremium += 0.5*dt*self.get_discount_factor(yieldcurveTenor, yieldcurveRate, tn)*(
                            self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1) - self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tn))
                return spread*(annuity+accruedPremium)
        else:  # When the cds maturity is beyond our current credit curve, we need to estimate the survival probability for payment beyond credit curve
            annuity = 0
            accruedPremium = 0
            N = int(cdsMaturity*num_premium_year)
            M = creditcurveTenor[max_time_index] * num_premium_year if max_time_index > 0 else 0

            for n in range(1, N+1):
                if n <= M:
                    tn = n/num_premium_year
                    tnm1 = (n-1)/num_premium_year
                    dt = 1.0 / num_premium_year

                    annuity += dt * self.get_discount_factor(yieldcurveTenor, yieldcurveRate, tn) * \
                    self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tn)
                    if(accruedPremiumFlag):
                        accruedPremium += 0.5*dt*self.get_discount_factor(yieldcurveTenor, yieldcurveRate,tn)*(
                            self.getSurvivalProbability(creditcurveTenor, creditcurveSP,tnm1) - \
                            self.getSurvivalProbability(creditcurveTenor, creditcurveSP,tn))
                else:
                    tn = n/num_premium_year
                    tnm1 = (n-1)/num_premium_year
                    tM = M / num_premium_year
                    dt = 1.0 / num_premium_year

                    survivalProbability_n = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tM) * \
                    np.exp(-h*(tn - tM))
                    survivalProbability_nm1 = 0
                    if tnm1 <= tM:
                        survivalProbability_nm1 = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1)
                    else:
                        survivalProbability_nm1 = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tM) * np.exp(-h*(tnm1 - tM))
                    annuity += dt * self.get_discount_factor(yieldcurveTenor, yieldcurveRate, tn)*survivalProbability_n
                    if accruedPremiumFlag:
                        accruedPremium += 0.5*dt * self.get_discount_factor(yieldcurveTenor, yieldcurveRate , tn)* \
                        (survivalProbability_nm1-survivalProbability_n)

            return spread*(annuity+accruedPremium)
        
    def calculateDefaultLeg(self, creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, 
                        cdsMaturity, num_default_year, recoveryRate, h):
        max_time_index = len(creditcurveTenor) - 1
        if max_time_index > 0 and cdsMaturity <= creditcurveTenor[max_time_index]:
            annuity = 0
            N = int(cdsMaturity * num_default_year)
            for n in range(1, N+1):
                tn = n / num_default_year
                tnm1 = (n-1) / num_default_year
                annuity += self.get_discount_factor(yieldcurveTenor, yieldcurveRate, tn)*(
                    self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1) - \
                    self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tn))
            return (1-recoveryRate)*annuity
        else:  
            # cdsMaturity > creditcurveTenor[max_time_index]
            annuity = 0
            N = int(cdsMaturity*num_default_year)
            M = creditcurveTenor[max_time_index] * num_default_year if max_time_index > 0 else 0

            for n in range(1, N+1):
                if n <= M:
                    tn = n / num_default_year
                    tnm1 = (n-1) / num_default_year
                    annuity += self.get_discount_factor(yieldcurveTenor, yieldcurveRate, tn)*(
                        self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1) - \
                        self.getSurvivalProbability(creditcurveTenor, creditcurveSP,tn))
                else:  # n > m
                    tM = M / num_default_year
                    tn = n / num_default_year
                    tnm1 = (n-1) / num_default_year

                    survivalProbability_n = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tM)*\
                                            np.exp(-h*(tn-tM))
                    if tnm1 <= tM:
                        survivalProbability_nm1 = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1)
                    else:
                        survivalProbability_nm1 = self.getSurvivalProbability(creditcurveTenor, creditcurveSP,  tM) * np.exp(-h*(tnm1 - tM))
                    annuity += self.get_discount_factor(yieldcurveTenor, yieldcurveRate, tn) * (survivalProbability_nm1 - 
                                                                                                survivalProbability_n)

            return (1-recoveryRate)*annuity
    def objfunFindHazardRate(self, h):
        # print(cdsMaturity)
        premLeg = self.calculatePremiumLeg(creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, cdsMaturity, premiumFrequency, 
                                     accruedPremium, spread,h)
        defaultLeg = self.calculateDefaultLeg(creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, cdsMaturity, defaultFrequency, 
                                     recoveryRate, h)
        return premLeg - defaultLeg
    
    def bootstrapCDSspread(self):
        yieldcurveRate = self.yieldcurve
        cdsSpreads = self.spread 
        premiumFrequency = self.premiumFrequency
        defaultFrequency = self.defaultFrequency
        
        accruedPremium = self.accruedPremium
        recoveryRate = self.recoveryRate
        
        yieldcurveLength = len(self.yieldcurveLength)
        cdsTenorsLength = len(self.cdsTenor)

        newcreditcurveLength = 0
        newcreditcurve = []
        survprob = [None]*cdsTenorsLength
        hazardRate = [None]*cdsTenorsLength
        global creditcurveSP 
        creditcurveSP = []
        global creditcurveTenor 
        creditcurveTenor = []
        for i in range(cdsTenorsLength):
            global cdsMaturity 
            cdsMaturity = cdsTenors[i]
            global spread
            spread = cdsSpreads[i]
            # print(cdsMaturity, spread)
            h = newton(self.objfunFindHazardRate, cdsSpreads[i])
            if h > 1:
                h = 1
            hazardRate[i] = h
            if i==0:
                survprob[i] = np.exp(-hazardRate[i]*cdsTenors[i])
            else:
                survprob[i] = survprob[i-1]*np.exp(-hazardRate[i]*(cdsTenors[i]-cdsTenors[i-1]))
            creditcurveTenor.append(cdsTenors[i])
            creditcurveSP.append(survprob[i])
        return hazardRate, survprob
        
    if __name__ == "__main__":
          # Define inputs of a cds contract
          yieldcurveTenor = [0.5, 1, 2, 3, 4, 5]
          yieldcurveRate = [0.01350, 0.01430, 0.0190, 0.02470, 0.02936, 0.03311]

          """
          This is to construct CDS, you can put in any value when getting the hazard curve
          """
          creditcurveTenor = [1, 2, 5]
          creditcurveSP = [99, 98, 95]

          cdsTenors = [0.25, 0.5,1, 2]
          cdsSpreads = [0.2443, 0.2766, 0.44, 0.5483]
          premiumFrequency = 4
          defaultFrequency = 12
          accruedPremium = True
          recoveryRate = 0.40
test_cds = cds(creditcurveSP, creditcurveTenor, yieldcurveRate, yieldcurveTenor,
             cdsTenors, premiumFrequency, defaultFrequency, accruedPremium, recoveryRate, cdsSpreads)
result = test_cds.bootstrapCDSspread()
