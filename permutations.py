import main as base
import numpy as np
# random change to test 
# also testing

def main():
    # print(reduce("ABABABAB"))

    for j in range(1, 17):
      print(j)
      l = perms(j)
      for i in range(len(l)):
          s = l[i]
          if s != reduce(s): continue
          n = base.h_str_to_mat(s)
          if np.array_equal(n, base.I3): print(s)

def reduce(s):
    result = s
    for i in range(len(s)-2):
        if s[i] == s[i+1] and s[i] == s[i+2]:
            sBefore = s[:i]
            sAfter = s[i+3:]
            result = sBefore + sAfter
            break
    s = result
    for i in range(len(s)-7):
      if s[i:i+8] == "ABABABAB" or s[i:i+8] == "BABABABA":
        result = s[:i] + s[i+8:]
    if result != s:
      return reduce(result)
    else:
      return result

def perms(length):
    if length == 1:
        return ["A", "B"]
    else:
        previous = perms(length-1)
        result = []
        for i in range(len(previous)):
            result.append(previous[i]+"A")
            result.append(previous[i]+"B")
        return result

if __name__ == "__main__":
    main()
