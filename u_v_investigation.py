import math
#looking just at u = -v
# for n in range (1, 2000):
#     m = math.sqrt(1296 + n**2)
#     mInt = int(m)
#     if m == mInt:
#         print(m, "      ", math.sqrt(36+m), "     ")

#still just looking at u = -v
# for u in range (9, 2000):
#     d = math.sqrt(u**2*(-72+u**2))
#     dInt = int(d)
#     if (d==dInt):
#         print(u)

threshold = 1e-10

#Looking more broadly at u and v
for u  in range (-10000,10000):
    if u != 0: #u can't be zero, otherwise many entries are undefined
        for v in range (-10000,10000):
            if v/u == v//u: # checking that a12 and a33 are integers
                entry32 = (-(u**2)+u*v-(v**2))/(u**2)
                entry32Int = int(entry32) 
                if abs(entry32 - entry32Int) <= threshold: #checking that a32 is an integer
                    t = 2*(u**2+v**2)*(u**2-u*v+v**2) #computing the value to tau
                    d = (-4)*(u**2)*(5+u)+(4)*(u)*(8+u)*(v)+(-20+(u)*(4+u))*(v**2)-4*(v**3) # computing the value of d
                    if d > 0: #d has to be positive for its square root to be an integer
                        dRoot = math.sqrt(d)
                        entry13 = u*(-(u**3)*v+4*u*(v**2)-2*(v**3)+(u**2)*(-6*v+dRoot))/t
                        entry13Int = int(entry13)
                        if abs(entry13-entry13Int) <= threshold: #checking that a13 is an integer
                            #at this point A is in SL3Z since a matrix of this form will always have det 1
                            b13 = u * (2*(u**3)+2*(v**3)-(u**2)*v*(2+v)+u*v*(-2*v + dRoot))/t
                            b13Int = int(b13)
                            if abs(b13 - b13Int) <= threshold: #checking that b13 is an integer
                                b21 = (-2*(v**2) + (u**2)*(2+v) + u *(4*v+ dRoot))/ (2* (u**2))
                                b21Int = int(b21)
                                if abs(b21Int - b21) <= threshold: #checking that b21 is an integer
                                    b31 = (-2*(u**3)-2*(v**3)+(u**2)*v*(2+v)+u*v*(2*v+dRoot))/(2*(u**3))
                                    b31Int = int(b31)
                                    if abs(b31 - b31Int) <= threshold: #checking that b31 is an integer
                                        det = (-1 + v/u)*(-v/u)-b13*b31
                                        if det == 1:
                                            print("Solution: ",u,v)
print("finished")