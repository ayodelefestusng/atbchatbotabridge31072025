# myproject/myapp/forms.py
from django import forms
from django.contrib.auth.forms import PasswordResetForm, SetPasswordForm
from .models import CustomUser

class RegistrationForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ["email", "full_name"]

    def clean_email(self):
        email = self.cleaned_data.get("email")
        if CustomUser.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email

class PasswordSetupForm(SetPasswordForm):
    pass

        
class PasswordChangeForm(forms.Form):
    old_password = forms.CharField(widget=forms.PasswordInput)
    new_password = forms.CharField(widget=forms.PasswordInput)
    

from django import forms
from .models import Prompt

class PromptForm(forms.ModelForm):
    class Meta:
        model = Prompt
        fields = ['summarize_prompt', 'sql_prompt', 'response_prompt']
        
        

from django import forms
from .models import Client

class ClientForm(forms.ModelForm):
    class Meta:
        model = Client
        fields = ["company_name", "phone_number", "address", "city", "website", "logo", "color_code", "state"]
    
    def clean_phone_number(self):
        phone = self.cleaned_data["phone_number"]
        if not (phone.startswith("07") or phone.startswith("08") or phone.startswith("09")) or len(phone) != 11:
            raise forms.ValidationError("Phone number must be 11 digits, starting with 0 and second digit between 7 and 9.")
        return phone