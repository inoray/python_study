'''
타이타닉 데이터 분석
생존자와 관련된 feature 들을 분석
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc


def title_mapping(x):
    #if x in set(['Capt', 'Col', 'Don', 'Major', 'Rev', 'Sir', 'Jonkheer']):
    if x.Title in set(['Don', 'Rev', 'Sir', 'Jonkheer']):
        return 'Mr'
    elif x.Title in set(['Lady', 'the Countess']):
        return 'Mrs'
    elif x.Title in set(['Mlle', 'Mme', 'Dona', 'Ms']):
        return "Miss"
    elif x.Title in set(['Major', 'Col', 'Capt']):
        return "Officer"
    elif x.Title == 'Dr' and x.Sex == 'female':
        return 'Mrs'
    elif x.Title == 'Dr' and x.Sex == 'male':
        return 'Mr'
    else:
        return x.Title


# plot에 한글 출력위한 폰트 설정
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

df_train = pd.read_csv('data/train.csv', header=0)

fig = plt.figure(figsize=(18, 12))
print(df_train['Survived'].size)
print(df_train['Survived'].value_counts())
print(df_train['Survived'][df_train['Sex'] == 'male'].value_counts())

ax1 = plt.subplot2grid((4, 4), (0, 0))
plt.title('생존/사망 비교')
df_train['Survived'].value_counts().sort_index().plot(kind='bar', ax=ax1, alpha=0.5)

ax2 = plt.subplot2grid((3, 4), (0, 1))
plt.title('pclass 생존/사망 비교')
df_train_class_by_survived = pd.DataFrame({'total': df_train['Pclass'].value_counts()
                                          , 'Non-Survivors': df_train['Pclass'][df_train['Survived'] == 0].value_counts()
                                          , 'Survivors': df_train['Pclass'][df_train['Survived'] == 1].value_counts()})
df_train_class_by_survived[['Non-Survivors', 'Survivors']].sort_index().plot(kind='bar', ax=ax2, alpha=0.5, rot=0)

ax3 = plt.subplot2grid((3, 4), (0, 2))
plt.title('성별 생존/사망 비교')
df_train_survived_by_sex = pd.DataFrame({'total': df_train['Sex'].value_counts()
                                        , 'Non-Survivors': df_train['Sex'][df_train['Survived'] == 0].value_counts()
                                        , 'Survivors': df_train['Sex'][df_train['Survived'] == 1].value_counts()})
df_train_survived_by_sex[['Non-Survivors', 'Survivors']].sort_index().plot(kind='bar', ax=ax3, alpha=0.5, rot=0)
print(df_train_survived_by_sex)

ax4 = plt.subplot2grid((3, 4), (0, 3))
plt.title('Embarked 생존/사망 비교')
df_train_Embarked_by_survived = pd.DataFrame({'total': df_train['Embarked'].value_counts()
                                             , 'Non-Survivors': df_train['Embarked'][df_train['Survived'] == 0].value_counts()
                                             , 'Survivors': df_train['Embarked'][df_train['Survived'] == 1].value_counts()})
df_train_Embarked_by_survived[['Non-Survivors', 'Survivors']].sort_index().plot(kind='bar', ax=ax4, alpha=0.5, rot=0)

ax10 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
hist_bins = len(df_train['Age'].unique())
#df_train['Age'].sort_index().plot(kind='hist', ax=ax10, alpha=0.5, bins=hist_bins)
plt.title('나이 생존/사망 비교')
gb_train_age_by_survived = df_train['Age'].groupby(df_train['Survived'])
df_train_age_by_survived = pd.DataFrame({'Non-Survivors': gb_train_age_by_survived.get_group(0)
                                         , 'Survivors': gb_train_age_by_survived.get_group(1)})
df_train_age_by_survived.plot(kind='hist', ax=ax10, alpha=0.5, bins=18, stacked=True)

ax11 = plt.subplot2grid((3, 4), (1, 2))
plt.title('SibSp 생존/사망 비교')
gb_train_sibsp_by_survived = df_train['SibSp'].groupby(df_train['Survived'])
df_train_sibsp_by_survived = pd.DataFrame({'Non-Survivors': gb_train_sibsp_by_survived.get_group(0)
                                           , 'Survivors': gb_train_sibsp_by_survived.get_group(1)})
df_train_sibsp_by_survived.plot(kind='hist', ax=ax11, alpha=0.5, bins=len(df_train['SibSp'].unique()), stacked=True)

ax12 = plt.subplot2grid((3, 4), (1, 3))
plt.title('Parch 생존/사망 비교')
gb_train_Parch_by_survived = df_train['Parch'].groupby(df_train['Survived'])
df_train_Parch_by_survived = pd.DataFrame({'Non-Survivors': gb_train_Parch_by_survived.get_group(0)
                                           , 'Survivors': gb_train_Parch_by_survived.get_group(1)})
df_train_Parch_by_survived.plot(kind='hist', ax=ax12, alpha=0.5, bins=len(df_train['Parch'].unique()), stacked=True)

ax13 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
plt.title('Fare 생존/사망 비교')
gb_train_fare_by_survived = df_train['Fare'].groupby(df_train['Survived'])
df_train_fare_by_survived = pd.DataFrame({'Non-Survivors': gb_train_fare_by_survived.get_group(0)
                                         , 'Survivors': gb_train_fare_by_survived.get_group(1)})
df_train_fare_by_survived.plot(kind='hist', ax=ax13, alpha=0.5, bins=12, stacked=True)


df_train['Title'] = df_train['Name'].apply(lambda x: (x.split(',')[1]).split('.')[0][1:])
df_train['Title'] = df_train.apply(title_mapping, axis=1)

ax14 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
plt.title('Title 생존/사망 비교')
df_train_Title_by_survived = pd.DataFrame({'total': df_train['Title'].value_counts()
                                             , 'Non-Survivors': df_train['Title'][df_train['Survived'] == 0].value_counts()
                                             , 'Survivors': df_train['Title'][df_train['Survived'] == 1].value_counts()})
df_train_Title_by_survived[['Non-Survivors', 'Survivors']].sort_index().plot(kind='bar', ax=ax14, alpha=0.5)
print(df_train['Title'].value_counts())
print(df_train_Title_by_survived)
plt.show()
