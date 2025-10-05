#!/usr/bin/env python3
"""
リアルタイムコラボレーション機能
WebSocketによるリアルタイム通信と共同作業機能
"""

import asyncio
import json
import websockets
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import uuid
import threading

logger = logging.getLogger(__name__)

@dataclass
class CollaborationSession:
    """コラボレーションセッション"""
    session_id: str
    created_at: datetime
    participants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_users: Set[str] = field(default_factory=set)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    chat_messages: List[Dict[str, Any]] = field(default_factory=list)
    max_participants: int = 50
    is_active: bool = True

class RealTimeCollaborationManager:
    """リアルタイムコラボレーションマネージャー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sessions: Dict[str, CollaborationSession] = {}
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 8765)
        self.server = None

    async def start_server(self):
        """WebSocketサーバーを起動"""
        self.server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port
        )

        logger.info(f"Collaboration server started on {self.host}:{self.port}")

    def create_session(self, session_id: str = None) -> str:
        """セッションを作成"""
        if not session_id:
            session_id = str(uuid.uuid4())

        if session_id not in self.sessions:
            self.sessions[session_id] = CollaborationSession(
                session_id=session_id,
                created_at=datetime.now()
            )

        return session_id

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """セッション情報を取得"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return {
            'session_id': session_id,
            'created_at': session.created_at.isoformat(),
            'active_users': list(session.active_users),
            'participant_count': len(session.participants)
        }

    async def _handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """WebSocket接続を処理"""
        connection_id = str(uuid.uuid4())
        self.websocket_connections[connection_id] = websocket

        try:
            async for message in websocket:
                data = json.loads(message)
                await self._process_message(connection_id, data)
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]

    async def _process_message(self, connection_id: str, data: Dict[str, Any]):
        """メッセージを処理"""
        message_type = data.get('type')

        if message_type == 'join_session':
            session_id = data.get('session_id', 'default')
            user_id = data.get('user_id', f'user_{connection_id[:8]}')

            if session_id not in self.sessions:
                self.create_session(session_id)

            session = self.sessions[session_id]
            session.participants[user_id] = {'connection_id': connection_id}
            session.active_users.add(user_id)

            # 参加をブロードキャスト
            await self._broadcast_to_session(session_id, {
                'type': 'user_joined',
                'user_id': user_id
            })

        elif message_type == 'chat_message':
            session_id = data.get('session_id', 'default')
            message = data.get('message', '')
            user_id = data.get('user_id', 'anonymous')

            if session_id in self.sessions:
                chat_msg = {
                    'user_id': user_id,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }

                self.sessions[session_id].chat_messages.append(chat_msg)

                # メッセージをブロードキャスト
                await self._broadcast_to_session(session_id, {
                    'type': 'chat_message',
                    'message': chat_msg
                })

    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """セッションにブロードキャスト"""
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]
        for user_id in session.active_users:
            participant = session.participants.get(user_id, {})
            connection_id = participant.get('connection_id')

            if connection_id and connection_id in self.websocket_connections:
                websocket = self.websocket_connections[connection_id]
                try:
                    await websocket.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")

# グローバルインスタンス
collaboration_manager = None

def initialize_collaboration(config: Dict[str, Any]):
    """コラボレーションを初期化"""
    global collaboration_manager

    collaboration_manager = RealTimeCollaborationManager(config)

    logger.info("Real-time collaboration initialized")

# 使用例
async def main():
    """メイン関数"""
    config = {
        'host': '0.0.0.0',
        'port': 8765
    }

    initialize_collaboration(config)

    # サーバーを起動
    await collaboration_manager.start_server()

    # セッションを作成
    session_id = collaboration_manager.create_session('demo_session')
    print(f"Session created: {session_id}")

    try:
        await collaboration_manager.server.wait_closed()
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())
