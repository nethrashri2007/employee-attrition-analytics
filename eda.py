import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/employee_attrition.csv")

# ==========================================
# 1. ATTRITION DISTRIBUTION
# ==========================================

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Attrition"
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.show()


# ==========================================
# 2. OVERTIME VS ATTRITION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)

plt.title("Overtime vs Employee Attrition")
plt.xlabel("Overtime")
plt.ylabel("Number of Employees")

plt.show()


# ==========================================
# 3. DEPARTMENT VS ATTRITION
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Department",
    hue="Attrition"
)

plt.title("Department vs Employee Attrition")
plt.xlabel("Department")
plt.ylabel("Number of Employees")

plt.xticks(rotation=20)

plt.show()


# ==========================================
# 4. JOB SATISFACTION VS ATTRITION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="JobSatisfaction",
    hue="Attrition"
)

plt.title("Job Satisfaction vs Employee Attrition")
plt.xlabel("Job Satisfaction")
plt.ylabel("Number of Employees")

plt.show()


# ==========================================
# 5. MONTHLY INCOME VS ATTRITION
# ==========================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title("Monthly Income vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")

plt.show()


# ==========================================
# 6. AGE VS ATTRITION
# ==========================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="Age"
)

plt.title("Age vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Age")

plt.show()


# ==========================================
# 7. DISTANCE FROM HOME VS ATTRITION
# ==========================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="DistanceFromHome"
)

plt.title("Distance From Home vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Distance From Home")

plt.show()


# ==========================================
# 8. YEARS SINCE LAST PROMOTION
# ==========================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="YearsSinceLastPromotion"
)

plt.title(
    "Years Since Last Promotion vs Employee Attrition"
)

plt.xlabel("Attrition")
plt.ylabel("Years Since Last Promotion")

plt.show()
# ==========================================
# ATTRITION RATE ANALYSIS
# ==========================================

print("\n" + "=" * 50)
print("ATTRITION RATE ANALYSIS")
print("=" * 50)


# ------------------------------------------
# Overall Attrition Rate
# ------------------------------------------

overall_rate = (
    df["Attrition"].eq("Yes").mean() * 100
)

print(
    f"\nOverall Attrition Rate: "
    f"{overall_rate:.2f}%"
)


# ------------------------------------------
# Overtime Attrition Rate
# ------------------------------------------

overtime_rate = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition Rate by Overtime:")
print(overtime_rate)


# ------------------------------------------
# Department Attrition Rate
# ------------------------------------------

department_rate = pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition Rate by Department:")
print(department_rate)


# ------------------------------------------
# Job Satisfaction Attrition Rate
# ------------------------------------------

satisfaction_rate = pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition Rate by Job Satisfaction:")
print(satisfaction_rate)


# ------------------------------------------
# Business Travel Attrition Rate
# ------------------------------------------

travel_rate = pd.crosstab(
    df["BusinessTravel"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition Rate by Business Travel:")
print(travel_rate)


# ------------------------------------------
# Marital Status Attrition Rate
# ------------------------------------------

marital_rate = pd.crosstab(
    df["MaritalStatus"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition Rate by Marital Status:")
print(marital_rate)