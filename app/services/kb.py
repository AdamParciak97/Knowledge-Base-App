from app.services.kb_base import KnowledgeBaseBase
from app.services.kb_builds import BuildMixin
from app.services.kb_documents import DocumentMixin


class KnowledgeBaseService(BuildMixin, DocumentMixin, KnowledgeBaseBase):
    pass


kb_service = KnowledgeBaseService()
