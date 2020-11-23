class Employee:
    numEmployee = 0
    raiseSalary = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + "." + last + "@email.com"
        self.pay = pay

        Employee.numEmployee += 1

    def fullname(self):
        return "{} {}".format(self.first, self.last)

    def applySalary(self):
        self.pay = int(self.pay * self.raiseSalary)

    @classmethod
    def set_raise_amt(cls, amount):
        cls.raiseSalary = amount

    @classmethod
    def from_string(cls, em_str):
        fir, la, p = em_str.split("-")
        return cls(fir, la, p)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True


Employee.set_raise_amt(1.05)

emp_1 = Employee("Corey", "Schafer", 50000)
emp_2 = Employee("Test", "emply", 60000)

emp_str_1 = "John-Doe-70000"
emp_str_2 = "mecha-smith-60000"
emp_str_3 = "Jane-Moller-90000"

f, l, pp = emp_str_1.split("-")
new_emp_1 = Employee(f, l, pp)
new_emp_2 = Employee.from_string(emp_str_2)

print(Employee.raiseSalary)
print(emp_1.raiseSalary)
print(emp_2.raiseSalary)

print(new_emp_1.email)
print(new_emp_1.pay)


import datetime

my_date = datetime.date(2016, 7, 10)
print(Employee.is_workday(my_date))
