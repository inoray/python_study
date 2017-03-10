import csv as csv
import numpy as np

csv_file_object = csv.reader(open('data/train.csv'))
next(csv_file_object)
data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

print(data)

number_passengers = np.size(data[0::, 1].astype(np.float))
number_survived = np.sum(data[0::, 1].astype(np.float))
proportion_survivors = number_survived / number_passengers
print("noPassengers = ", number_passengers)
print("noSurvived = ", number_survived)
print("proportion_survivors = ", proportion_survivors)

women_only_stats = data[0::, 4] == "female" # This finds where all
                                           # gender 변수에서
                                           # the elements in the gender
                                           # 'female'값을 가진
                                           # column that equals “female”
                                           # 요소들만 찾기
men_only_stats = data[0::, 4] != "female"   # This finds where all the
                                           # gender 변수에서
                                           # elements do not equal
                                           # female이 아닌 변수들만

# Using the index from above we select the females and males separately
# 위에서 남성과 여성을 분리했던 것을 인덱싱 한다.
women_onboard = data[women_only_stats, 1].astype(np.float)
# women_onboard에 여성중 생존자인 경우를 찾아 float 형태로 바꿔줌
men_onboard = data[men_only_stats, 1].astype(np.float)
# men_onboard에 남성중 생존자인 경우를 찾아 float형태로 바꿔줌
# Then we finds the proportions of them that survived
# 그러면 우리는 살아남은 사람들의 비율을 알 수 있다. (남 , 여)
proportion_women_survived = \
                       np.sum(women_onboard) / np.size(women_onboard)
# 여성의 살아 남은 비율
proportion_men_survived = \
                       np.sum(men_onboard) / np.size(men_onboard)
# 남성의 살아 남은 비율

# and then print it out
# 프린팅 해보자.
print('Proportion of women who survived is %s' % proportion_women_survived)
print('Proportion of men who survived is %s' % proportion_men_survived)

test_file = open('test.csv')
test_file_object = csv.reader(test_file)
next(test_file_object)

prediction_file = open('genderbasemodel.csv', 'w',  newline='')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerID", "Survived"])

for row in test_file_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0], '1'])
    else:
        prediction_file_object.writerow([row[0], '0'])
test_file.close()
prediction_file.close()


