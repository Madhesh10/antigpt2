# chat/models.py
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to="documents/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.title or self.file.name} ({self.user})"

class ChatThread(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="threads")
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Thread {self.id} - {self.user}"

class Message(models.Model):
    thread = models.ForeignKey(ChatThread, on_delete=models.CASCADE, related_name="messages")
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    content = models.TextField()
    is_bot = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("created_at",)

    def __str__(self):
        prefix = "Bot" if self.is_bot else f"User:{self.user}"
        return f"{prefix} - {self.content[:60]}"
