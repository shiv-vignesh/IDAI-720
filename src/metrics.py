import numpy as np

class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    def acc(self):
        # Accuracy
        return 1.0-np.sum(np.abs(np.array(self.y) - np.array(self.y_pred)))/len(self.y)

    def eod(self, s):
        # Equal Opportunity Difference
        # s: list/array of binary values 1/0, sensitive attribute
        # return: EOD = TPR(s=1)-TPR(s=0)
        # TPR = #(y=1, y_pred=1) / #(y=1)
        # Write your code below:

        target_bool = np.array(self.y) == 1
        predicted_bool = np.array(self.y_pred) == 1 
        senstive_bool_0 = np.array(s) == 0
        senstive_bool_1 = np.array(s) == 1

        tpr1 = np.sum(target_bool & predicted_bool & senstive_bool_1)/np.sum(target_bool & senstive_bool_1)
        tpr0 = np.sum(target_bool & predicted_bool & senstive_bool_0)/np.sum(target_bool & senstive_bool_0)

        return tpr1-tpr0

    def aod(self, s):
        # Average Odds Difference
        # s: list/array of binary values 1/0, sensitive attribute
        # return: AOD = (TPR(s=1)-TPR(s=0)+FPR(s=1)-FPR(s=0))/2
        # TPR = #(y=1, y_pred=1) / #(y=1)
        # FPR = #(y=0, y_pred=1) / #(y=0)
        # Write your code below:

        target_bool_0 = np.array(self.y) == 0
        target_bool_1 = np.array(self.y) == 1
        predicted_bool = np.array(self.y_pred) == 1 
        senstive_bool_0 = np.array(s) == 0
        senstive_bool_1 = np.array(s) == 1        

        tpr1 = np.sum(target_bool_1 & predicted_bool & senstive_bool_1)/np.sum(target_bool_1 & senstive_bool_1)
        tpr0 = np.sum(target_bool_1 & predicted_bool & senstive_bool_0)/np.sum(target_bool_1 & senstive_bool_0)

        fp_r1 = np.sum(target_bool_0 & predicted_bool & senstive_bool_1)/np.sum(target_bool_0 & senstive_bool_1) 
        fp_r0 = np.sum(target_bool_0 & predicted_bool & senstive_bool_0)/np.sum(target_bool_0 & senstive_bool_0) 

        return (tpr1 - tpr0 + fp_r1 - fp_r0)/2.0

    def spd(self, s):
        # Statistical Parity Difference
        # s: list/array of binary values 1/0, sensitive attribute
        # return: SPD = |PR(s=1)-PR(s=0)|
        # PR = #(y_pred=1) / #y
        # Write your code below:

        target_bool_1 = np.array(self.y) == 1

        senstive_bool_0 = np.array(s) == 0
        senstive_bool_1 = np.array(s) == 1        

        pr1 = np.sum(target_bool_1 & senstive_bool_1)/np.sum(senstive_bool_1)
        pr0 = np.sum(target_bool_1 & senstive_bool_0)/np.sum(senstive_bool_0)

        return pr1 - pr0


