# class MovingTotal:
#     def __init__(self):
#         self.values = []
#         self.sums = set()
#         self.running_sum = 0

#     def append(self, numbers):
#         """
#         :param numbers: (list) The list of numbers.
#         """
#         for number in numbers:
#             if len(self.values) >= 3:
#                 self.running_sum -= self.values.pop(0)
#             self.values.append(number)
#             self.running_sum += number
#             if len(self.values) == 3:
#                 self.sums.add(self.running_sum)

#     def contains(self, total):
#         """
#         :param total: (int) The total to check for.
#         :returns: (bool) If MovingTotal contains the total.
#         """
#         return total in self.sums

# if __name__ == "__main__":
#     movingtotal = MovingTotal()
    
#     movingtotal.append([1, 2, 3, 4])
#     print(movingtotal.contains(6))
#     print(movingtotal.contains(9))
#     print(movingtotal.contains(12))
#     print(movingtotal.contains(7))
    
#     movingtotal.append([5])
#     print(movingtotal.contains(6))
#     print(movingtotal.contains(9))
#     print(movingtotal.contains(12))
#     print(movingtotal.contains(7))

def nth_lowest_selling(sales, n):
    """
    :param sales: (list) List of book sales.
    :param n: (int) The n-th lowest selling element the function should return.
    :returns: (int) The n-th lowest selling book id in the book sales list.
    """
    counts = {}
    for sale in sales:
        counts[sale] = counts.get(sale, 0) + 1

    unique_counts = [(value, count) for value, count in counts.items()]
    unique_counts.sort(key=lambda x: x[1])

    return unique_counts[n - 1][0]

if __name__ == "__main__":
    print(nth_lowest_selling([5, 4, 2, 2, 1, 5, 4, 3, 2, 5, 4, 3, 5, 4, 5], 2))

