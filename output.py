import csv

with open('result.csv','a',newline='') as fs:
    a = csv.writer(fs,delimiter=",")
    data = []
    data = ["filename", "face"]
    a.writerow(data)


