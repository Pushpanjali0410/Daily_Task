from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class CustomerSupportData(BaseModel):
    name: str = Field(..., description="Customer's full name")
    email: EmailStr = Field(..., description="Customer's email address")
    order_id: str = Field(..., description="Order ID from the support message")
    issue_type: str = Field(..., description="Type of issue reported")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Rahul Sharma",
                "email": "rahul.sharma@gmail.com",
                "order_id": "ORD-45678",
                "issue_type": "Payment Failed"
            }
        }
