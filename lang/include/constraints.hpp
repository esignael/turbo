
I. x + y <= c

 ==> (tell)

(ub(x) + ub(y) > c) => x <- [lb(x)..c - lb(y)]
(ub(x) + ub(y) > c) => y <- [lb(y)..c - lb(x)]

 ==> (not)

-x - y <= -c

 ==> (ask)

ask(x) /\ ask(y) /\ ub(x) + ub(y) <= c


II. c1 \/ c2

  ==> (tell)

ask(not(c1)) => c2
ask(not(c2)) => c1

  ==> (not)

not(c1) /\ not(c2)

  ==> (ask)

ask(c1)
ask(c2)

III. c1 /\ c2 (similar to c1 /\ c2)

IV. b <=> c1

  ==> (tell)

lb(b) = 1 /\ ub(b) = 1 => c1
lb(b) = 0 /\ ub(b) = 0 => not(c1)
ask(c1) => b = 1
ask(not(c1)) => b = 0

  ==> (not)

not(b) <=> c1

  ==> (ask)

b = 1 /\ ask(c1)
b = 0 /\ ask(not(c1))

V. x1c1 + x2c2 + ... + xNcN <= c

  ==> (tell)

true => potential <- ub(x1) * c1 + ... + ub(xN) * cN
true => slack <- c - (lb(x1) * c1 + lb(x2) * c2 + ... + lb(xN) * cN)
lb(x1) != ub(x1) /\ c1 > slack => x1 <- [0..0]
...
lb(xN) != ub(xN) /\ cN > slack => xN <- [0..0]
slack < 0 => false

  ==> (not)

x1c1 + x2c2 + ... + xNcN > c

  ==> (ask)

ask(x1) /\ ... /\ ask(xN) /\ potential <= max

VI. Variable x

  ==> (ask)

lb(x) <= ub(x)

