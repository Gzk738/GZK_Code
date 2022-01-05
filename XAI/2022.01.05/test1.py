import time
temp = 0
while(True):
    temp +=1
    time.sleep(0.5)
    print(temp, "second")
    if temp>9999:
        temp = 0