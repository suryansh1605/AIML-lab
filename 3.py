import csv
with open("C:\\Users\\Surya\\OneDrive\\Desktop\\New folder\\trainingexamples.csv") as f:
 csv_file=csv.reader(f)
 data=list(csv_file)
 specific=data[3][:-1]
 general=[['?' for i in range(len(specific))] for j in range(len(specific))]
 for i in data:
     if i[-1]=="Y":
         for j in range(len(specific)):
             if i[j]!=specific[j]:
                 specific[j]='?'
                 general[j][j]='?'
     elif i[-1]=="N":
         for j in range(len(specific)):
             if i[j]!=specific[j]:
                 general[j][j]=specific[j]
             else :
                 general[j][j]='?'
print("\n step",data.index(i)+1," of candidate elemination algorithm")
print(specific)
print(general)
gh=[]
for i in general:
    for j in i:
        if j!='?':
            gh.append(i)
            break
print("Final Specific hypothesis : \n",specific)
print("Final general hypothesis : \n",gh)