# frontend/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import ensure_csrf_cookie

def signup_page(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        if username and password:
            User.objects.create_user(username=username, password=password)
            return redirect("/login/")
    return render(request, "frontend/signup.html")

def login_page(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(username=username, password=password)
        if user:
            login(request, user)
            return redirect("/chat/")
    return render(request, "frontend/login.html")

def logout_view(request):
    logout(request)
    return redirect("/login/")

@ensure_csrf_cookie
@login_required
def chat_page(request):
    return render(request, "frontend/chat.html")

# --- new homepage view ---
def home_page(request):
    """
    Public homepage. Shows links to signup/login/chat.
    If you prefer to redirect Home to /chat/ for logged-in users,
    you can add logic here to redirect based on request.user.is_authenticated.
    """
    # Example: redirect logged-in users to chat automatically
    if request.user.is_authenticated:
        return redirect("/chat/")
    return render(request, "frontend/home.html")
