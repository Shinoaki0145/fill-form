"""Silence segmentation and serialized native recognition."""
import asyncio
from collections import deque
import math

class AudioRunner:
    """Chốt đoạn theo audio clock; ASR không được kéo dài ranh giới đoạn."""
    def __init__(self,session,backend,vad,config,segment_silence=1.0):
        from .turns import TurnBuffer
        if not math.isfinite(segment_silence) or not 0<segment_silence<config.max_turn:
            raise ValueError('segment_silence phải >0 và nhỏ hơn max_turn')
        self.session=session;self.backend=backend;self.vad=vad;self.turn=TurnBuffer(config)
        self.segment_silence=segment_silence
        self.asr_task=None;self.snapshot=None;self.turn_generation=None
        self.pending=deque()

    @property
    def busy(self):
        return self.turn.active or bool(self.pending) or self.asr_task is not None

    async def consume(self,frame):
        turn=self.turn;s=self.session
        if frame is not None and turn.feed(frame,self.vad(frame)):
            s.speech_started();self.turn_generation=s.generation
        key=(turn.turn_id,turn.revision)
        if self.asr_task and self.asr_task.done():
            task=self.asr_task;self.asr_task=None
            snap=self.snapshot
            try: text,seconds=task.result()
            except Exception as exc:
                if not getattr(s,'recover_asr_error',False):raise
                s.emit('asr_failed',turn_id=snap.turn_id,error_type=type(exc).__name__)
                text,seconds='',0.
            snap_key=(snap.turn_id,snap.revision)
            if self.pending and snap_key==self.pending[0]['key']:
                self.pending[0]['text']=text
                s.emit('asr_segment_ready',turn_id=snap.turn_id,asr_ms=round(seconds*1000,1))
            elif turn.active and snap_key==key:
                turn.set_text(snap.turn_id,snap.revision,text)
                s.emit('transcript_partial',text=text,asr_ms=round(seconds*1000,1))
            else: s.emit('stale_asr_discarded',turn_id=snap.turn_id)

        # Đóng băng audio ngay khi đủ im lặng, không chờ ASR hoàn tất.
        if turn.active and (turn.silence>=self.segment_silence or turn.at_limit):
            if len(self.pending)>=8:
                raise RuntimeError('ASR không theo kịp: đã có 8 đoạn audio chờ, dừng để tránh mất dữ liệu')
            snapshot=turn.snapshot()
            cached=turn.text if turn.text_revision==turn.revision else None
            self.pending.append(dict(snapshot=snapshot,key=key,generation=self.turn_generation,text=cached))
            s.emit('segment_closed',turn_id=turn.turn_id,utterance_generation=self.turn_generation,
                   audio_time=round(turn.clock/16000,3),silence_seconds=round(turn.silence,3),
                   silence_limit=self.segment_silence,reason='max_turn' if turn.at_limit else 'silence',
                   asr_pending=cached is None)
            turn.reset()

        # Giữ thứ tự transcript, kể cả khi người dùng đã bắt đầu đoạn mới trước khi ASR xong.
        while self.pending and self.pending[0]['text'] is not None:
            item=self.pending.popleft()
            s.submit(item['text'],utterance_generation=item['generation'],turn_id=item['snapshot'].turn_id)

        # Một native ASR duy nhất; ưu tiên đoạn đã chốt, rồi snapshot tạm mới nhất.
        if self.asr_task is None:
            if self.pending:
                self.snapshot=self.pending[0]['snapshot']
            elif turn.needs_asr():
                self.snapshot=turn.snapshot()
            else: return
            self.asr_task=asyncio.create_task(asyncio.to_thread(self.backend.recognize,self.snapshot.audio))

    async def close(self):
        # Không giả vờ có thể hủy native thread; đợi công việc đang chạy rồi mới giải phóng backend.
        if self.asr_task: await asyncio.shield(self.asr_task)
