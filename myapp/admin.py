from django.contrib import admin

# Register your models here.
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import *

class CustomUserAdmin(UserAdmin):
    # These fields will be displayed in the admin list view
    list_display = ('email', 'full_name', 'is_staff', 'is_active')
    # These fields will be displayed when viewing/editing a user
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('full_name',)}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
        ('Facial Data', {'fields': ('face_data', 'face_image')}), # Add face fields
    )
    # Fields that can be used to search for users
    search_fields = ('email', 'full_name')
    # Fields that can be used to filter users
    ordering = ('email',)
    # Use the custom add form
    # add_form = CustomUserCreationForm # If you had a custom creation form

# Unregister the default User model if it was registered
# from django.contrib.auth.models import User
# admin.site.unregister(User) # Uncomment if you're not extending AbstractUser for some reason

admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(Prompt)  # Registers Prompt model with default ModelAdmin
admin.site.register(Prompt7) 