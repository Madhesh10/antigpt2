# chat/serializers.py
from rest_framework import serializers
from .models import Document, ChatThread, Message

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id','user','title','file','uploaded_at','processed']
        read_only_fields = ('user','uploaded_at','processed')

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id','thread','user','content','is_bot','created_at']
        read_only_fields = ('is_bot','created_at')

class ChatThreadSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = ChatThread
        fields = ['id','user','title','created_at','messages']
        read_only_fields = ('user','created_at','messages')
