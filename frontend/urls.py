# frontend/urls.py
from django.urls import path
from .views import signup_page, login_page, chat_page, logout_view, home_page

urlpatterns = [
    # Home page (root)
    path("", home_page, name="home"),

    # Auth & chat pages
    path("signup/", signup_page, name="signup"),
    path("login/", login_page, name="login"),
    path("logout/", logout_view, name="logout"),
    path("chat/", chat_page, name="chat"),
]
