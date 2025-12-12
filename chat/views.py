# chat/views.py
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Document, ChatThread, Message
from .serializers import DocumentSerializer, ChatThreadSerializer, MessageSerializer
from .ingest import ingest_document_file, answer_question

class DocumentViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user).order_by("-uploaded_at")

    def perform_create(self, serializer):
        doc = serializer.save(user=self.request.user)
        # For now run ingestion synchronously. For production move to background worker.
        try:
            ingest_document_file(doc)
        except Exception as e:
            # mark processed to avoid re-run if something failed (optional)
            doc.processed = False
            doc.save()

class ChatThreadViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ChatThreadSerializer

    def get_queryset(self):
        return ChatThread.objects.filter(user=self.request.user).order_by("-created_at")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["post"])
    def ask(self, request, pk=None):
        thread = self.get_object()
        question = request.data.get("question", "").strip()
        if not question:
            return Response({"error":"question is required"}, status=status.HTTP_400_BAD_REQUEST)
        # save user message
        user_msg = Message.objects.create(thread=thread, user=request.user, content=question, is_bot=False)
        # get answer from RAG
        try:
            answer = answer_question(request.user, question)
        except Exception as e:
            answer = "Error generating answer: " + str(e)
        bot_msg = Message.objects.create(thread=thread, content=answer, is_bot=True)
        return Response({"answer": answer, "user_message_id": user_msg.id, "bot_message_id": bot_msg.id})
