class StringQueue:
    def __init__(self, max_size: int):
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("队列大小必须是正整数")
        self.max_size = max_size
        self.queue = []
    
    def enqueue(self, item: str):
        if not isinstance(item, str):
            raise TypeError("队列只接受字符串类型")
        if self.is_full():
            removed_item = self.queue.pop(0)
        self.queue.append(item)
    
    def dequeue(self) -> str:
        if self.is_empty():
            raise ValueError("队列为空")
        return self.queue.pop(0)
    
    def is_empty(self) -> bool:
        return len(self.queue) == 0
    
    def is_full(self) -> bool:
        return len(self.queue) >= self.max_size
    
    def size(self) -> int:
        return len(self.queue)
    
    def __str__(self) -> str:
        return f"StringQueue({self.queue})"
    
    def concat_with_spaces(self) -> str:
        """将队列中所有字符串用序号和中文标点连接成新字符串
        返回:
            str: 格式为"1，xxx。2，xxx。"的字符串，空队列返回空字符串
        """
        if not self.queue:
            return ''
            
        # 生成带序号的字符串列表（序号从1开始）
        numbered = [f"{i}，{s}" for i, s in enumerate(self.queue, start=1)]
        # 用句号+空格连接，最后补一个句号
        return '。 '.join(numbered) + '。'
    