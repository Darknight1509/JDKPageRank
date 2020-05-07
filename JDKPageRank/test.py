with open('classname') as file_object:
    contents = file_object.read()
list=contents.split()
print(list[:20])


