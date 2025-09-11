import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def proc (dataF):
    df = dataF
    a = np.linspace(df['age'].min()-3, df['age'].max()+3, 70)
    s = np.linspace(df['sex'].min(), df['sex'].max(), 2)
    cpt = np.linspace(df['chest pain type'].min(), df['chest pain type'].max(), 4)
    rbs = np.linspace(df['resting bp s'].min(), df['resting bp s'].max()+2, 201)
    cltr = np.linspace(df['cholesterol'].min(), df['cholesterol'].max()+3, 605)
    fbs = np.linspace(df['fasting blood sugar'].min(), df['fasting blood sugar'].max(), 2)
    re = np.linspace(df['resting ecg'].min(), df['resting ecg'].max(), 3)
    mhr = np.linspace(df['max heart rate'].min()-3, df['max heart rate'].max()+3, 201)
    ea = np.linspace(df['exercise angina'].min(), df['exercise angina'].max(), 2)
    op = np.linspace(df['oldpeak'].min()-1, df['oldpeak'].max()+1, 80)
    ss = np.linspace(df['ST slope'].min(), df['ST slope'].max(), 4)
    #--------------declare input
    age = ctrl.Antecedent(a, 'age')
    sex = ctrl.Antecedent(s, 'sex')
    chestPainType = ctrl.Antecedent(cpt, 'cp_type')
    restingBpS = ctrl.Antecedent(rbs, 'rt_bps')
    cholesterol = ctrl.Antecedent(cltr, 'c_rol')
    fastingBloodSugar = ctrl.Antecedent(fbs, 'ft_bluds')
    restingEcg = ctrl.Antecedent(re, 'rt_ecg')
    maxHeartRate = ctrl.Antecedent(mhr, 'max_hrate')
    exerciseAngina = ctrl.Antecedent(ea, 'ex_angina')
    oldpeak = ctrl.Antecedent(op, 'oldpeak')
    stSlope = ctrl.Antecedent(ss, 'st_slope')
    #--------------declare output
    target = ctrl.Consequent(np.arange(0, 1, 0.1), 'target')
    #--------------set fuzzy
    age['Youth'] = fuzz.gaussmf(age.universe, 25, 15)
    age['Medium'] = fuzz.gaussmf(age.universe, 50, 15)
    age['Old'] = fuzz.gaussmf(age.universe, 70, 15)

    sex['Male'] = fuzz.trimf(sex.universe, [1, 1, 1])
    sex['Female'] = fuzz.trimf(sex.universe, [0, 0, 0])

    chestPainType['Weak'] = fuzz.trimf(chestPainType.universe, [1, 1, 1])
    chestPainType['Normal'] = fuzz.trimf(chestPainType.universe, [2, 2, 2])
    chestPainType['Hard'] = fuzz.trimf(chestPainType.universe, [3, 3, 3])
    chestPainType['Too Hard'] = fuzz.trimf(chestPainType.universe, [4, 4, 4])

    restingBpS['Slow'] = fuzz.trimf(restingBpS.universe, [0, 0, 110])
    restingBpS['Normal'] = fuzz.trimf(restingBpS.universe, [100, 110, 125])
    restingBpS['Fast'] = fuzz.trimf(restingBpS.universe, [120, 130, 145])
    restingBpS['Too Fast'] = fuzz.trimf(restingBpS.universe, [140, 200, 200])

    cholesterol['Normal'] = fuzz.trimf(cholesterol.universe, [0, 0, 200])
    cholesterol['High'] = fuzz.trimf(cholesterol.universe, [190, 220, 240])
    cholesterol['Too High'] = fuzz.trimf(cholesterol.universe, [235, 600, 600])

    fastingBloodSugar['True'] = fuzz.trimf(fastingBloodSugar.universe, [1, 1, 1])
    fastingBloodSugar['False'] = fuzz.trimf(fastingBloodSugar.universe, [0, 0, 0])

    restingEcg['Normal'] = fuzz.trimf(restingEcg.universe, [0, 0, 0])
    restingEcg['Abnormal'] = fuzz.trimf(restingEcg.universe, [1, 1, 1])
    restingEcg['Dangerous'] = fuzz.trimf(restingEcg.universe, [2, 2, 2])

    maxHeartRate['Low'] = fuzz.trimf(maxHeartRate.universe, [50, 80, 120])
    maxHeartRate['Medium'] = fuzz.trimf(maxHeartRate.universe, [110, 130, 150])
    maxHeartRate['High'] = fuzz.trimf(maxHeartRate.universe, [140, 160, 200])

    exerciseAngina['True'] = fuzz.trimf(exerciseAngina.universe, [1, 1, 1])
    exerciseAngina['False'] = fuzz.trimf(exerciseAngina.universe, [0, 0, 0])

    oldpeak['Normal'] = fuzz.trimf(oldpeak.universe, [0, 0, 0.2])
    oldpeak['Abnormal'] = fuzz.trimf(oldpeak.universe, [0.1, 0.5, 1.4])
    oldpeak['Dangerous'] = fuzz.trimf(oldpeak.universe, [1, 6, 6])

    stSlope['Up'] = fuzz.trimf(stSlope.universe, [0, 0, 0])
    stSlope['Flat'] = fuzz.trimf(stSlope.universe, [1, 1, 1])
    stSlope['Down'] = fuzz.trimf(stSlope.universe, [2, 2, 2])
