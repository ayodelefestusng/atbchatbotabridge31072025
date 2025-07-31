# myapp/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.utils.translation import gettext_lazy as _
from django.conf import settings # Still needed if CustomUser remains primary for other features

class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(email, password, **extra_fields)

class CustomUser(AbstractUser):
    # Remove username field completely
    username = None
    
    # Personal Info
    full_name = models.CharField(_('full name'), max_length=255)
    email = models.EmailField(_('email address'), unique=True)
    
    # Biometric Fields
    face_data = models.TextField(
        _('face data'),
        blank=True, 
        null=True, 
        help_text="JSON or path to stored facial biometric data/embeddings"
    )
    face_image = models.ImageField(
        _('face image'),
        upload_to='face_images/', 
        blank=True, 
        null=True, 
        help_text="Stored image for facial recognition reference"
    )

    # Authentication Fields
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['full_name']
    
    objects = CustomUserManager()

    def __str__(self):
        return self.email

    class Meta:
        verbose_name = _('User')
        verbose_name_plural = _('Users')
        
# --- END CustomUser ---

class Post(models.Model):
    title = models.CharField(max_length=150)
    content = models.TextField()
    # Keep this tied to CustomUser if posts are user-specific,
    # otherwise, if posts are general and not linked to a user, remove this field.
    # For now, assuming posts might still be created by an authenticated user/admin.
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

class Conversation(models.Model):
    # Removed user ForeignKey. Using session_id instead.
    session_id = models.CharField(max_length=255, unique=True, db_index=True) # Unique per session for active conversations
    started_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Conversation {self.id} (Session: {self.session_id})"

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    text = models.TextField()
    is_user = models.BooleanField()
    timestamp = models.DateTimeField(auto_now_add=True)
    attachment = models.FileField(upload_to='chat_attachments/', null=True, blank=True)

    class Meta:
        ordering = ['timestamp']

class LLM(models.Model):
    currecy_choices =[('NGN','Naira'),('GBP','Pound'),('EURO','Euro'),('USD','usd'),('DEDI','DEDI')]
    currency  = models.CharField( max_length=50 ,choices=currecy_choices, default='NGN')
    naira = models.DecimalField( max_digits=50, decimal_places=2,default= 1)
    pound =models.DecimalField( max_digits=50, decimal_places=2,default= 1)
    euro =models.DecimalField( max_digits=50, decimal_places=2,default= 1)
    usd = models.DecimalField( max_digits=50, decimal_places=2,default= 1)
    cedi = models.DecimalField( max_digits=50, decimal_places=2,default= 1)
    
    def __str__(self):
        return self.currency

class SessionData(models.Model):
    # This model already uses session_id
    session_id = models.IntegerField(unique=True) # Consider if this should be CharField for session keys
    sentiment = models.IntegerField()
    ticket = models.JSONField()  # Stores items as JSON

    def __str__(self):
        return f"Session {self.session_id}"

class Sentiment(models.Model):
    # This model already uses session_id
    session_id = models.IntegerField(unique=True) # Consider if this should be CharField for session keys
    sentimentperQuest = models.IntegerField()
    question = models.CharField(max_length=150)

    def __str__(self):
        return f"Session {self.session_id}"

class Insight(models.Model):
    # Removed user ForeignKey. Using session_id instead.
    session_id = models.CharField(max_length=255, db_index=True) # Not unique because multiple insights per session
    question = models.TextField(default="")
    answer = models.TextField()
    sentiment = models.IntegerField(default=0, null=True, blank=True)
    ticket = models.JSONField(blank=True, null=True)  # For storing list of strings
    source = models.JSONField(blank=True, null=True)  # For storing list of strings
    summary = models.TextField()
    sum_sentiment = models.CharField(max_length=140, default="")
    sum_ticket = models.JSONField(blank=True, null=True) 
    sum_source = models.JSONField(blank=True, null=True)  
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True) 
    
    class Meta:
        # To get the latest insight for a session, order by updated_at or created_at
        ordering = ['-updated_at'] 

    def __str__(self):
        return f"Insight {self.id} | Session: {self.session_id} | Created: {self.created_at} | Updated: {self.updated_at}"

class Checkpoint(models.Model):
    thread_id = models.TextField()
    checkpoint_ns = models.TextField()
    data = models.BinaryField()
    

class Ticket(models.Model):
    # Removed user ForeignKey. Using session_id instead.
    session_id = models.CharField(max_length=255, unique=True, db_index=True) # Unique per session for active tickets
    started_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Ticket {self.id} (Session: {self.session_id})"
    
    
    
    
# Copilot for faker 
class NigerianName(models.Model):
    first_name = models.CharField(max_length=50)  # Remove `unique=True`
    last_name = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"

class Branch(models.Model):
    branch_id = models.CharField(max_length=20, unique=True)
    branch_name = models.CharField(max_length=100)
    state_address = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.branch_name} ({self.branch_id})"

class Customer(models.Model):
    customer_id = models.CharField(max_length=20, unique=True)
    first_name = models.ForeignKey("NigerianName", on_delete=models.SET_NULL, null=True, related_name="first_names")
    last_name = models.ForeignKey("NigerianName", on_delete=models.SET_NULL, null=True, related_name="last_names")
    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=11, unique=True)
    account_number = models.CharField(max_length=10, unique=True)

    gender = models.CharField(max_length=10, choices=[("male", "Male"), ("female", "Female")])
    city_of_residence = models.CharField(max_length=100)
    state_of_residence = models.CharField(max_length=100)
    nationality = models.CharField(max_length=50, default="Nigeria")
    occupation = models.CharField(max_length=100)
    date_of_birth = models.DateField()

    branch = models.ForeignKey("Branch", on_delete=models.SET_NULL, null=True)  # Customer's registered branch

    def __str__(self):
        return f"{self.first_name} {self.last_name} - {self.account_number}"

    def clean(self):
        """Custom validation for phone numbers and account numbers."""
        if not self.phone_number.startswith("0") or self.phone_number[1] not in "6789":
            raise ValueError("Phone number must start with '0' and second digit must be between 6 and 9.")
        if len(self.account_number) != 10 or not self.account_number.isdigit():
            raise ValueError("Account number must be exactly 10 digits.")

class Transaction(models.Model):
    TRANSACTION_TYPES = [
        ("deposit", "Deposit"),
        ("withdrawal", "Withdrawal"),
        ("transfer", "Transfer"),
        ("airtime", "Airtime Purchase"),
        ("loan", "Loan Disbursement"),
        ("bill_payment", "Bill Payment"),
        ("balance_enquiry", "Balance Enquiry"),
    ]

    TRANSACTION_CHANNELS = [
        ("atm", "ATM"),
        ("pos", "POS"),
        ("branch", "Branch"),
        ("web", "Web"),
        ("mobile", "Mobile"),
    ]

    transaction_id = models.CharField(max_length=30, unique=True)
    customer = models.ForeignKey("Customer", on_delete=models.CASCADE)
    transaction_type = models.CharField(max_length=20, choices=TRANSACTION_TYPES)
    amount = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    service_charge = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    transaction_channel = models.CharField(max_length=10, choices=TRANSACTION_CHANNELS)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.transaction_type} via {self.transaction_channel} - {self.amount}"


class LoanReport(models.Model):
    customer = models.ForeignKey("Customer", on_delete=models.CASCADE)
    loan_account_number = models.CharField(max_length=20, unique=True)
    amount_collected = models.DecimalField(max_digits=12, decimal_places=2)
    date_loan_booked = models.DateField()
    last_repayment_date = models.DateField(null=True, blank=True)
    loan_balance = models.DecimalField(max_digits=12, decimal_places=2)

    branch_booked = models.ForeignKey("Branch", on_delete=models.SET_NULL, null=True)  # Branch where loan was processed

    def __str__(self):
        return f"Loan {self.loan_account_number} - Balance: {self.loan_balance}"


class ComplianceRecord(models.Model):
    record_id = models.CharField(max_length=30, unique=True)
    transaction = models.ForeignKey("Transaction", on_delete=models.CASCADE)
    compliance_status = models.CharField(max_length=50, choices=[("passed", "Passed"), ("flagged", "Flagged")])
    audit_notes = models.TextField(blank=True)
    checked_by = models.CharField(max_length=255)
    checked_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Compliance {self.record_id} - {self.compliance_status}"
    
class BranchPerformance(models.Model):
    branch = models.ForeignKey("Branch", on_delete=models.CASCADE)
    total_customers = models.PositiveIntegerField()
    total_transactions = models.PositiveIntegerField()
    revenue_generated = models.DecimalField(max_digits=12, decimal_places=2)
    report_date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.branch.branch_name} - {self.report_date}"                    
    

class Employee(models.Model):
    employee_id = models.CharField(max_length=20, unique=True)
    full_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    department = models.CharField(max_length=100)
    role = models.CharField(max_length=100)
    salary = models.DecimalField(max_digits=10, decimal_places=2)
    hire_date = models.DateField()

    def __str__(self):
        return self.full_name
    
    
from django.db import models


class Prompt(models.Model):
    summarize_prompt = models.TextField()
    sql_prompt = models.TextField()
    response_prompt = models.TextField()
    sql_response_prompt = models.TextField(default="")
    CHATBOT_MODEL_CHOICES = [
        ("gpt", "gpt"),
        ('gemini', "gemini"),
        ("deepseek", "deepseek"),
        ("groq", "groq"),
    ]
    chatbot_model = models.CharField( max_length=100, choices=CHATBOT_MODEL_CHOICES,default="gemini" )
    variance_prompt = models.TextField(default="")
    google_model = models.CharField( max_length=100, default="gemini-2.0-flash" )
    # views_model = models.CharField( max_length=100, default="gemini-2.0-flash" )
    
    
    def __str__(self):
        return f"Prompt {self.id}"

CHATBOT_MODEL_CHOICES = [
            ("gpt", "gpt"),
            ('gemini', "gemini"),
            ("deepseek", "deepseek"),
            ("groq", "groq"),
            ]
        
class Prompt7(models.Model):
    """Model definition for MODELNAME."""

    # TODO: Define fields here
    summarize_prompt = models.TextField()
    # ay= models.TextField()
    sql_prompt = models.TextField()
    response_prompt = models.TextField()
    sql_response_prompt = models.TextField(default="")

    chatbot_model= models.CharField( max_length=100, choices=CHATBOT_MODEL_CHOICES,default="gemini")
    variance_prompt = models.TextField(default="")
    google_model = models.CharField( max_length=100, default="gemini-2.0-flash" )
    views_model = models.CharField( max_length=100, default="gemini-2.0-flash" )

    def __str__(self):
        """Unicode representation of MODELNAME."""
        return f"Prompt {self.id}"
    
    


    
    
from django.db import models
import random
import string

# Define choices for Nigerian States
NIGERIAN_STATES = [
    ("Abia", "Abia"), ("Adamawa", "Adamawa"), ("Akwa Ibom", "Akwa Ibom"), ("Anambra", "Anambra"),
    ("Bauchi", "Bauchi"), ("Bayelsa", "Bayelsa"), ("Benue", "Benue"), ("Borno", "Borno"),
    ("Cross River", "Cross River"), ("Delta", "Delta"), ("Ebonyi", "Ebonyi"), ("Edo", "Edo"),
    ("Ekiti", "Ekiti"), ("Enugu", "Enugu"), ("FCT", "FCT"), ("Gombe", "Gombe"),
    ("Imo", "Imo"), ("Jigawa", "Jigawa"), ("Kaduna", "Kaduna"), ("Kano", "Kano"),
    ("Katsina", "Katsina"), ("Kebbi", "Kebbi"), ("Kogi", "Kogi"), ("Kwara", "Kwara"),
    ("Lagos", "Lagos"), ("Nasarawa", "Nasarawa"), ("Niger", "Niger"), ("Ogun", "Ogun"),
    ("Ondo", "Ondo"), ("Osun", "Osun"), ("Oyo", "Oyo"), ("Plateau", "Plateau"),
    ("Rivers", "Rivers"), ("Sokoto", "Sokoto"), ("Taraba", "Taraba"), ("Yobe", "Yobe"), ("Zamfara", "Zamfara"),
]

# Function to generate unique 4-character alphanumeric client ID
def generate_client_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

class Client(models.Model):
    company_name = models.CharField(max_length=255)
    phone_number = models.CharField(
        max_length=11,
        unique=True,
        help_text="Must be 11 digits, starting with 0 and second digit between 7 and 9"
    )
    address = models.TextField()
    city = models.CharField(max_length=100)
    # website = models.URLField(blank=True, null=True)
    website = models.CharField(max_length=255, blank=True, null=True)
    logo = models.ImageField(upload_to="client_logos/", blank=True, null=True)
    color_code = models.CharField(max_length=7, help_text="Hex color code (e.g., #FF5733)")
    state = models.CharField(max_length=50, choices=NIGERIAN_STATES)
    client_id = models.CharField(max_length=4, unique=True, default=generate_client_id)

    def __str__(self):
        return self.company_name

class Ayo(models.Model):
    name = models.CharField(max_length=255)
    def __str__(self):
        return self.name
