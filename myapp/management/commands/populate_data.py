import random
import datetime
from django.core.management.base import BaseCommand
from myapp.models import NigerianName, Branch, Customer, Transaction, LoanReport, ComplianceRecord, BranchPerformance, Employee

NIGERIAN_FIRST_NAMES = ["Chinedu", "Emeka", "Abiola", "Uche", "Adebayo", "Ngozi", "Mary", "Tunde", "Hauwa", "Fatima"]
NIGERIAN_LAST_NAMES = ["Okonkwo", "Adeyemi", "Mohammed", "Ibrahim", "Eze", "Osagie", "Umeh", "Bassey", "Olufemi", "Balogun"]
NIGERIAN_BRANCHES = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Kano", "Owerri", "Kaduna", "Enugu", "Abeokuta", "Warri"]

# START_YEAR = 2010
START_YEAR = 2025

TODAY = datetime.date.today()

def random_date(start_year=START_YEAR, end_date=TODAY):
    """Generate a random date between a start year and today."""
    start_date = datetime.date(start_year, 1, 1)
    time_between_dates = (end_date - start_date).days
    random_days = random.randint(0, time_between_dates)
    return start_date + datetime.timedelta(days=random_days)

class Command(BaseCommand):
    help = "Populate database with sample data"

    def handle(self, *args, **kwargs):
        self.populate_nigerian_names()
        self.populate_branches()
        self.populate_customers()
        self.populate_transactions()
        self.populate_loans()
        self.populate_compliance_records()
        self.populate_branch_performance()
        self.populate_employees()

        self.stdout.write(self.style.SUCCESS("Database populated successfully!"))

    def populate_nigerian_names(self):
        for _ in range(400):
            first_name = random.choice(NIGERIAN_FIRST_NAMES)
            last_name = random.choice(NIGERIAN_LAST_NAMES)
        
        # Check if name already exists
            if not NigerianName.objects.filter(first_name=first_name, last_name=last_name).exists():
             NigerianName.objects.create(first_name=first_name, last_name=last_name)
    def populate_branches(self):
        for _ in range(150):
            branch_id = f"BR-{random.randint(1000, 9999)}"
            branch_name = random.choice(NIGERIAN_BRANCHES)
            state_address = random.choice(NIGERIAN_BRANCHES)

            # Ensure unique branch_id
            if not Branch.objects.filter(branch_id=branch_id).exists():
                Branch.objects.create(branch_id=branch_id, branch_name=branch_name, state_address=state_address)

    def generate_unique_customer_id(self):
        """Ensure customer_id uniqueness."""
        customer_id = f"CUST-{random.randint(100000, 999999)}"
        while Customer.objects.filter(customer_id=customer_id).exists():
            customer_id = f"CUST-{random.randint(100000, 999999)}"
        return customer_id


    def generate_unique_phone(self):
        """Generate a unique phone number to avoid IntegrityError."""
        phone_number = f"0{random.choice(['6', '7', '8', '9'])}{random.randint(10000000, 99999999)}"

        # Ensure uniqueness before returning
        while Customer.objects.filter(phone_number=phone_number).exists():
            phone_number = f"0{random.choice(['6', '7', '8', '9'])}{random.randint(10000000, 99999999)}"

        return phone_number

    def generate_unique_email(self):
        """Generate a unique email to avoid IntegrityError."""
        email = f"user{random.randint(100000, 999999)}@example.com"
        
        # Ensure uniqueness before returning
        while Customer.objects.filter(email=email).exists():
            email = f"user{random.randint(100000, 999999)}@example.com"
        
        return email



    def generate_unique_account(self):
        """Generate a unique 10-digit account number."""
        account_number = str(random.randint(1000000000, 9999999999))  # Ensure 10 digits

        # Ensure uniqueness before returning
        while Customer.objects.filter(account_number=account_number).exists():
            account_number = str(random.randint(1000000000, 9999999999))

        return account_number
    
    
    def generate_unique_transaction_id(self):
        """Ensure transaction_id uniqueness."""
        transaction_id = f"TX-{random.randint(100000, 999999)}"
        
        while Transaction.objects.filter(transaction_id=transaction_id).exists():
            transaction_id = f"TX-{random.randint(100000, 999999)}"

        return transaction_id
    
    def populate_customers(self):
        for _ in range(2000):
            Customer.objects.create(
                customer_id=self.generate_unique_customer_id(),  # Ensure uniqueness
                first_name=NigerianName.objects.order_by("?").first(),
                last_name=NigerianName.objects.order_by("?").first(),
                email=self.generate_unique_email(),  # Call fixed method
                phone_number=self.generate_unique_phone(),
                account_number=self.generate_unique_account(),
                branch=Branch.objects.order_by("?").first(),
                date_of_birth=random_date(start_year=1970)
            )
            
            
        
        
        
        
        
    def generate_unique_transaction_id(self):
        """Generate a unique transaction ID."""
        transaction_id = f"TX-{random.randint(100000, 999999)}"
        
        while Transaction.objects.filter(transaction_id=transaction_id).exists():
            transaction_id = f"TX-{random.randint(100000, 999999)}"  # Keep regenerating until unique

        return transaction_id
            
        
        
        
        
        
    def populate_transactions(self):
        TRANSACTION_TYPES = ["deposit", "withdrawal", "transfer", "airtime", "loan", "bill_payment", "balance_enquiry"]
        TRANSACTION_CHANNELS = ["atm", "pos", "branch", "web", "mobile"]

        for _ in range(40000):
            Transaction.objects.create(
                transaction_id=self.generate_unique_transaction_id(),  # Fix: Ensure uniqueness
                customer=Customer.objects.order_by("?").first(),
                transaction_type=random.choice(TRANSACTION_TYPES),
                amount=random.uniform(100, 50000),
                service_charge=random.uniform(1, 500),
                transaction_channel=random.choice(TRANSACTION_CHANNELS),
                timestamp=random_date()
            )



    def populate_loans(self):
        for _ in range(600):
            LoanReport.objects.create(
                customer=Customer.objects.order_by("?").first(),
                loan_account_number=f"LN-{random.randint(100000, 999999)}",
                amount_collected=random.uniform(50000, 5000000),
                date_loan_booked=random_date(),
                last_repayment_date=random_date(),
                loan_balance=random.uniform(1000, 1000000),
                branch_booked=Branch.objects.order_by("?").first()
            )

    def populate_compliance_records(self):
        COMPLIANCE_STATUSES = ["passed", "flagged"]

        for _ in range(240):
            ComplianceRecord.objects.create(
                record_id=self.generate_unique_compliance_record_id(),  # Fix: Ensure uniqueness
                transaction=Transaction.objects.order_by("?").first(),
                compliance_status=random.choice(COMPLIANCE_STATUSES),
                audit_notes="System-generated audit check.",
                checked_by=random.choice(["Admin", "Audit Team"]),
                checked_date=random_date()
            )

    def populate_branch_performance(self):
        for _ in range(1000):
            BranchPerformance.objects.create(
                branch=Branch.objects.order_by("?").first(),
                total_customers=random.randint(500, 5000),
                total_transactions=random.randint(1000, 50000),
                revenue_generated=random.uniform(100000, 5000000),
                report_date=random_date()
            )


    def generate_unique_employee_email(self):
        """Generate a unique employee email."""
        email = f"employee{random.randint(1000, 99999)}@bank.com"

        while Employee.objects.filter(email=email).exists():
            email = f"employee{random.randint(1000, 99999)}@bank.com"

        return email

    def populate_employees(self):
        DEPARTMENTS = ["HR", "Finance", "Operations", "Compliance", "IT", "Customer Service"]
        ROLES = ["Manager", "Analyst", "Officer", "Clerk", "Supervisor"]

        for _ in range(800):
            Employee.objects.create(
                employee_id=f"EMP-{random.randint(1000, 9999)}",
                full_name=f"{random.choice(NIGERIAN_FIRST_NAMES)} {random.choice(NIGERIAN_LAST_NAMES)}",
                email=self.generate_unique_employee_email(),  # Fix: Ensure uniqueness
                department=random.choice(DEPARTMENTS),
                role=random.choice(ROLES),
                salary=random.uniform(50000, 500000),
                hire_date=random_date()
            )



# python manage.py populate_data