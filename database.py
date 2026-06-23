from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional
from datetime import datetime

DATABASE_URL = "sqlite:///./appointments.db"
engine = create_engine(DATABASE_URL, echo=False)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    phone: str = Field(unique=True, index=True)
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    date: str        # YYYY-MM-DD
    time: str        # HH:MM AM/PM
    status: str = "booked"   # booked | cancelled
    created_at: datetime = Field(default_factory=datetime.utcnow)


def create_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
