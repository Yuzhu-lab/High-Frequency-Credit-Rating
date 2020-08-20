import numpy as np
import pandas as pd
from scipy.optimize import bisect
from scipy.optimize import newton

class bond:
    def __init__(self, creditcurve, creditcurveLength, yieldcurve, yieldcurveLength,
                 BondTenor, CouponFrequency, CouponRate, defaultFrequency, recoveryRate, price, Par):
        self.creditcurve = creditcurve
        self.creditcurveLength = creditcurveLength
        self.yieldcurve = yieldcurve
        self.yieldcurveLength = yieldcurveLength
        self.BondTenor = BondTenor
        self.CouponFrequency = CouponFrequency
        self.CouponRate = CouponRate 
        self.defaultFrequency = defaultFrequency
        self.recoveryRate = recoveryRate
        self.price = price
        self.FaceValue = Par
        
    def get_discount_factor(self,t):
        yieldcurveTenor = self.yieldcurveLength
        yieldcurveRate = self.yieldcurve
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
                    if h > 1:
                        h = 1
                    result = creditcurveSP[i] * np.exp(-(t-creditcurveTenor[i])*h)
        return result
    
    def calculateCVA(self, creditcurve, creditcurveTenor,BondTenor, h):
        max_time_index = len(creditcurveTenor) - 1
        CVA = 0
        num_default_year = self.defaultFrequency
        N = int(BondTenor * num_default_year)
#         dt = 1.0 / num_default_year
        if max_time_index > 0 and BondTenor <= creditcurveTenor[max_time_index]:    
            for n in range(1, N+1):
                tn = n / num_default_year
                tnm1 = (n-1) / num_default_year
                t = 0.5 * (tn+tnm1)
                Exposure_t = self.get_exposure(t, BondTenor)
                CVA += self.get_discount_factor(t)*(
                    self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1) - \
                    self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tn)) * Exposure_t * (1 - self.recoveryRate)
            return CVA
        else:  
            # When the bond maturity is beyond our current credit curve, we need to estimate the survival probability for payment beyond credit curve
            M = creditcurveTenor[max_time_index] * num_default_year if max_time_index > 0 else 0
            for n in range(1, N+1):
                if n <= M:
                    tn = n/num_default_year
                    tnm1 = (n-1)/num_default_year
                    t = 0.5 * (tn+tnm1)
                    Exposure_t = self.get_exposure(t, BondTenor)
                    CVA += self.get_discount_factor(t) *( \
                    self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1) - \
                    self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tn)) * Exposure_t * (1 - self.recoveryRate)
                else:
                    tn = n/num_default_year
                    tnm1 = (n-1)/num_default_year
                    t = 0.5 * (tn+tnm1)
                    tM = M/num_default_year
                    Exposure_t = self.get_exposure(t, BondTenor)
#                     print("Exposure at ", t,"is ", Exposure_t)
                    survivalProbability_n = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tM) * \
                    np.exp(-h*(tn - tM))
                    
#                     survivalProbability_nm1 = 0
                    if tnm1 <= tM:
                        survivalProbability_nm1 = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1)
                    else:
                        survivalProbability_nm1 = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tM) * np.exp(-h*(tnm1 - tM))
                                                                                                                            
                    CVA += self.get_discount_factor(t)*(survivalProbability_nm1-survivalProbability_n)* Exposure_t * (1 - self.recoveryRate)
            return CVA
    
    def objfunFindHazardRate(self, h, creditcurveSP, creditcurveTenor, BondMaturity):
        VND = self.calculateCFND(BondMaturity)
        CVA = self.calculateCVA(creditcurveSP, creditcurveTenor, BondMaturity, h)
#         print("creditcurveSP, creditcurveTenor, BondMaturity, h", creditcurveSP, creditcurveTenor, BondMaturity, h)
#         print("VND", VND)
#         print("CVA", CVA)
        return VND - CVA            
        
    def calculateCFND(self,BondTenor):
        N = int(self.CouponFrequency * BondTenor) 
        Discounted_CF = 0
        for n in range(1, N+1):
            tn = n / self.CouponFrequency
#                 tnm1 = (n-1) / num_premium_year
#                 dt = 1.0 / num_premium_year
#             print("t is ", tn)
#             print(self.get_discount_factor(tn))
            Discounted_CF += self.get_discount_factor(tn)*(self.CouponRate/self.CouponFrequency)*self.FaceValue
            if n == N:
                Discounted_CF += self.get_discount_factor(tn)*self.FaceValue
        return Discounted_CF   
    
    def bootstrapBondPrice(self):
        yieldcurveRate = self.yieldcurve
        Price = self.price
        CouponFrequency = self.CouponFrequency
        defaultFrequency = self.defaultFrequency
        recoveryRate = self.recoveryRate
        yieldcurveLength = len(self.yieldcurveLength)
        BondTenorsLength = len(self.BondTenor)

        newcreditcurveLength = 0
        newcreditcurve = []
        survprob = [None]*BondTenorsLength
        hazardRate = [None]*BondTenorsLength
        global creditcurveSP 
        creditcurveSP = []
        global creditcurveTenor 
        creditcurveTenor = []
        for i in range(BondTenorsLength):
            global BondMaturity 
            BondMaturity = self.BondTenor[i]
            global price
            price = Price[i]
            print("price", price)
            def f(x, creditcurveSP, creditcurveTenor, BondMaturity):
#                 print("hazard_rate_test", x)
#                 print(x,BondMaturity, creditcurveSP, creditcurveTenor)
#                 print("price_test",self.objfunFindHazardRate(x,BondMaturity, creditcurveSP, creditcurveTenor))
#                 print(self.objfunFindHazardRate(x, [], [], BondMaturity) - price)
                return self.objfunFindHazardRate(x, creditcurveSP, creditcurveTenor, BondMaturity) - price
            h = bisect(f, 0, 1, args=(creditcurveSP, creditcurveTenor, BondMaturity))
            if h > 1:
                h = 1
            hazardRate[i] = h
            if i==0:
                survprob[i] = np.exp(-hazardRate[i]*self.BondTenor[i])
            else:
                survprob[i] = survprob[i-1]*np.exp(-hazardRate[i]*(self.BondTenor[i]-self.BondTenor[i-1]))
            creditcurveTenor.append(self.BondTenor[i])
            creditcurveSP.append(survprob[i])
        return hazardRate, survprob
    
    def get_exposure(self, t, BondTenor):
        partition_coupon = 1/self.CouponFrequency
#         partition_default = 1/self.defaultFrequency
        coupon_date = [i for i in np.arange(partition_coupon,BondTenor + partition_coupon,partition_coupon)]
        coupon_PMT = [self.FaceValue*self.CouponRate/self.CouponFrequency] \
        *(self.CouponFrequency * BondTenor-1)+[self.FaceValue \
                                                      *self.CouponRate/self.CouponFrequency+self.FaceValue]
        D = {coupon_date[i]:coupon_PMT[i] for i in range(0, len(coupon_PMT))}
        exposure = 0
        N = len(D)
        flag = False
        for x in range(0, N-1):
            if t < coupon_date[0]:
                flag = True
                exposure += coupon_PMT[0] * self.get_discount_factor(coupon_date[0] - t)
            elif coupon_date[x]<=t and coupon_date[x+1]>t:
                flag = True
            if flag == True:
                exposure += coupon_PMT[x+1] * self.get_discount_factor(coupon_date[x+1] - t)
        return exposure
if __name__ == "__main__":
    # Define inputs of a cds contract
    yieldcurveTenor = [1/12,2/12,3/12,6/12,1,2,3,5,7,10,20,30]
    yieldcurveRate = [0.0007,0.0009,0.0011,0.0012,0.0013,0.0014,0.0017,0.0029,0.0048,0.0068,0.0120,0.0142]
    
    """
    This is to construct Bond, you can put in any value when getting the hazard curve
    """
    creditcurveTenor = [0, 1, 3, 5, 7, 10, 5.5]
    creditcurveSP = [1, 0.90000, 0.80000, 0.50000, 0.20000, 0.1, 0.0]
    
    BondTenors = [1,2,3]
    price = [101, 103, 105]
    CouponFrequency = 4
    defaultFrequency = 1
    recoveryRate = 0.40
    CouponRate = 0.05
    par = 100

test_bond = bond(creditcurveSP, creditcurveTenor, yieldcurveRate, yieldcurveTenor,
             BondTenors, CouponFrequency,CouponRate, defaultFrequency, recoveryRate, price, par)
result = test_bond.bootstrapBondPrice()
