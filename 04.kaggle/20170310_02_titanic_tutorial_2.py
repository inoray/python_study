import csv as csv
import numpy as np

csv_file_object = csv.reader(open('data/train.csv'))
next(csv_file_object)
data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)

print(data)

fare_ceiling = 40

data2 = data[0::, 9].astype(np.float)
#print(data2)

data3 = data2 >= fare_ceiling
#print(data3)

data[data3, 9] = fare_ceiling - 1.0
#print(data)

"""
modifiedDatafile = open("modifiedTrain.csv", 'w', newline='')
modifiedDatafileObject = csv.writer(modifiedDatafile)
for row in data:
    modifiedDatafileObject.writerow(row)
modifiedDatafile.close()
"""

fare_bracket_size = 10
number_of_price_brackets = int(fare_ceiling / fare_bracket_size)

number_of_classes = len(np.unique(data[0::, 2]))

survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

print(survival_table)

for i in range(number_of_classes):  # loop through each class
    # 각 클래스안에서 반복문 , xrange는 이 자체 객체를 리턴, 리스트를 만들지 않고 작동
    for j in range(number_of_price_brackets):  # loop through each price bin

        women_only_stats = data[                            # which element
                                (data[0::, 4] == "female")
                                & (data[0::, 2].astype(np.float)   # and was ith class
                                == i + 1)
                                & (data[0:, 9].astype(np.float)    # was greater
                                >= j * fare_bracket_size)     # than this bin
                                & (data[0:, 9].astype(np.float)    # and less than
                                < (j + 1) * fare_bracket_size)  # the next bin
                                , 1]  # in the 2nd col
        # women_only_stats 변수에는 여성이면서 등급별, 티켓 가격 범주별로 생존 유무 반환
        survival_table[0][i][j] = np.sum(women_only_stats.astype(np.float))


        men_only_stats = data[                              # Which element
                                (data[0::, 4] != "female")     # is a male
                                & (data[0::, 2].astype(np.float)   # and was ith class
                                == i + 1)
                                & (data[0:, 9].astype(np.float)    # was greater
                                >= j * fare_bracket_size)     # than this bin
                                & (data[0:, 9].astype(np.float)    # and less than
                                < (j + 1) * fare_bracket_size)  # the next bin
                                , 1]
                                # men_only_stats 변수에는 남성이면서 등급별, 티켓 가격 범주별로 생존 유무 반환
        survival_table[1][i][j] = np.sum(men_only_stats.astype(np.float))
        #print(women_only_stats)
        #print(men_only_stats)
print(survival_table)


