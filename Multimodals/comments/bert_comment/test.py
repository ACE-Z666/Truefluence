from bert_comment.run_engagement import analyze_engagement
print(analyze_engagement(10000, 50, 5))      # fake → LOW
print(analyze_engagement(10000, 900, 120))   # real → HIGH
print(analyze_engagement(50000, 200, 10))    # fake → LOW
print(analyze_engagement(50000, 5000, 600))  # real → HIGH
print(analyze_engagement(1, 1, 0))     # fake → MEDIUM