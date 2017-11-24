import sys

filename = sys.argv[1]
n = eval(sys.argv[2])
percent = eval(sys.argv[3])

ngram_set = set()
dict = {}

def compute_threshold(values, t):
  values.sort(reverse=True)
  csum = 0
  total_sum = sum(values)
  ret = -1
  for i in xrange(len(values)):
    csum += values[i]
    if csum > t * total_sum:
      ret = max(ret, values[i])
      return ret
  return ret

with open(filename) as f:
  for line in f.readlines():
    line = line.strip()
    sents1, sents2 = line.split('\t')
    ngrams1 = [sents1[i:i+n] for i in xrange(len(sents1)-n)]
    ngrams2 = [sents2[i:i+n] for i in xrange(len(sents2)-n)]
    #print sents1, ngrams1
    #print sents2, ngrams2
    for ng in ngrams1:
      if ng in dict:
        dict[ng] = dict[ng]+1
      else:
        dict[ng] = 1
    for ng in ngrams2:
      if ng in dict:
        dict[ng] = dict[ng]+1
      else:
        dict[ng] = 1
thr = compute_threshold(dict.values(), percent)
high_freq = filter(lambda i: dict[i]>thr, dict.keys())
print len(high_freq)
for h in high_freq:
  print h, dict[h]

