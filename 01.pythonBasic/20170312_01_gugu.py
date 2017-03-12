"""
print("- 구구단")

def gugu(n):
    for i in range(10):
        print("%d x %d = %d" % (n, i, (n*i)))

print("출력하고자 하는 단을 입력하세요: ")
nn = input()
gugu(int(nn))
"""

print("- 1000 미만의 자연수에서 3의 배수와 5의 배수의 총합을 구하라")


def getBeasu(r, n):
    result = []
    for i in r:
        if divmod(i, n)[1] == 0:
            result.append(i)
    return result


print("숫자를 입력하세요: ")
number = int(input())
while number < 0 or number > 1000:
    print("0보다 크고 1000보다 작은 값을 입력하세요")
    number = int(input())

rng = range(number)
beasu3 = getBeasu(rng, 3)
beasu5 = getBeasu(rng, 5)

print("3의 배수")
print(beasu3)
print("5의 배수")
print(beasu5)

print("두 리스트 결합")
sumList = beasu3 + beasu5
sumList.sort()
print(sumList)

print("중복제거")
sumList = list(set(sumList))
print(sumList)

print("모든 숫자 합계")
total = sum(sumList)
print(total)
