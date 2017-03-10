import numpy as np

print("- 1로 채워진 3*2 행렬 생성")
a = np.ones((3, 2))
print(a)

print("- 0로 채워진 행렬")
b = np.zeros((3, 2))
print(b)

print("- 직접 행렬 생성")
c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(c)
print(c.shape)
print(c.dtype)

print("- 형변환")
c_float = c.astype(np.float64)
print(c_float)
print(c_float.dtype)

print("- dtype 속성으로 직접 데이터형 지정")
d = np.array(["1.25", "0.99", "44", "11"], dtype=np.string_)
print(d)

print("- d를 float으로 변환")
d_float = d.astype(float)
print(d_float)

# 색인연산
# 출처: http://pinkwink.kr/716
print("\n*** 색인연산 ***")
print("- name array")
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "joe"])
print(names)

print("- dataName array")
dataNames = np.random.randn(7, 4)
print(dataNames)

print("- nameBob = names == 'Bob' 수행")
nameBob = names == 'Bob'
print(nameBob)

print("- dataNames[nameBob] 수행")
dataOfBob = dataNames[nameBob]
print(dataOfBob)

print("- Bob | Will 수행")
nameBobOrWill = (names == 'Bob') | (names == 'Will')
print(nameBobOrWill)
dataOfBobOrWill = dataNames [nameBobOrWill]
print(dataOfBobOrWill)

print("- dataNames[dataNames < 0] = 0")
dataNames[dataNames < 0] = 0
print(dataNames)

print("- dataNames[names != 'Joe'] = 7")
dataNames[names != 'Joe'] = 7
print(dataNames)
