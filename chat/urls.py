# chat/urls.py
from rest_framework.routers import DefaultRouter
from .views import DocumentViewSet, ChatThreadViewSet

router = DefaultRouter()
router.register(r"documents", DocumentViewSet, basename="documents")
router.register(r"threads", ChatThreadViewSet, basename="threads")

urlpatterns = router.urls
