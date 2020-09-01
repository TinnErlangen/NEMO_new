### geparkt: code für EPO numbers log und/oder equalizing epoch counts between conditions ###

#equalize number of epochs per condition by random dropping excess epochs - does not take tones(r1-s2) into account
    len_diff = len(pos.events)-len(neg.events)
    if len_diff > 0:
        drops = random.sample(range(len(pos.events)),k=len_diff)
        drops.sort(reverse=True)
        for drop in drops:
            pos.drop([drop])
    if len_diff < 0:
        drops = random.sample(range(len(neg.events)),k=abs(len_diff))
        drops.sort(reverse=True)
        for drop in drops:
            neg.drop([drop])

#prepare a logfile to track the sound/condition numbers in equalizing the epochs per condition
logfile = meg_dir+"epo_equalizing_v1.txt"
with open(logfile,"w") as file:
    file.write("Subject\tNeg_r1\tNeg_r2\tNeg_s1\tNeg_s2\tPos_r1\tPos_r2\tPos_s1\tPos_s2\n")

    file.write("{s}\t{n1}\t{n2}\t{n3}\t{n4}\t{p1}\t{p2}\t{p3}\t{p4}\n".format(s=meg,n1=len(neg['r1']),n2=len(neg['r2']),n3=len(neg['s1']),n4=len(neg['s2']),
                   p1=len(pos['r1']),p2=len(pos['r2']),p3=len(pos['s1']),p4=len(pos['s2'])))
