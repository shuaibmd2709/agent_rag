from src.guard import check

context = (
    "Non-compliance with the prohibition of the AI practices referred to in "
    "Article 5 shall be subject to administrative fines of up to EUR 35 000 000 "
    "or, if the offender is an undertaking, up to 7 % of its total worldwide "
    "annual turnover for the preceding financial year, whichever is higher."
)

faithful = context
cases = {
    "faithful": faithful,
    "invented number": faithful.replace("35 000 000", "40 000 000"),
    "invented sentence": faithful
    + "\nProviders must also register every system with a national ombudsman "
    "within fourteen days of deployment in any member state.",
}

for name, answer in cases.items():
    r = check(answer, context)
    print(f"{name:18} flagged={r['flagged']}  numbers={r['unsupported_numbers']}  "
          f"weak_lines={len(r['weak_lines'])}")