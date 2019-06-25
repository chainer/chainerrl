import time


if __name__=="__main__":
    print("Script started!!!")
    for i in range(200):
        print("Sleeping for",(200-i)*60," seconds more!")
        time.sleep(60)